import os
import sys

# Force global UTF-8 mode on Windows
if sys.platform == "win32" and os.environ.get("PYTHONUTF8") != "1":
    import subprocess

    os.environ["PYTHONUTF8"] = "1"
    result = subprocess.run([sys.executable] + sys.argv, env=os.environ)
    sys.exit(result.returncode)

import hashlib
import io
import math
import re
import time
import yaml
import logging
import threading
import asyncio
from pathlib import Path
from collections import defaultdict, Counter

# Fix for Windows: avoid "Event loop is closed" with ProactorEventLoop
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import nest_asyncio

# Apply nest_asyncio only when needed (conflicts with uvicorn)
_NEST_ASYNCIO_APPLIED = False


def _ensure_nest_asyncio():
    global _NEST_ASYNCIO_APPLIED
    if not _NEST_ASYNCIO_APPLIED:
        nest_asyncio.apply()
        _NEST_ASYNCIO_APPLIED = True

# Force UTF-8 on stdout/stderr
try:
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )
except Exception:
    pass

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from llama_index.core import (
    PropertyGraphIndex,
    StorageContext,
    SimpleDirectoryReader,
    Settings,
    PromptTemplate,
    load_index_from_storage,
)
from llama_index.core.indices.property_graph.transformations import (
    SimpleLLMPathExtractor,
    ImplicitPathExtractor,
)
from llama_index.core.indices.property_graph import (
    LLMSynonymRetriever,
    VectorContextRetriever,
    CustomPGRetriever,
)
from llama_index.core.schema import NodeWithScore, TextNode, Document
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# Multilingual module
from lang_config import (
    tokenize,
    detect_language,
    extract_date_tokens,
    infer_relation,
    ALL_TEMPORAL_KEYWORDS,
    FALLBACK_RELATION,
)

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_NAME = "gpt-oss:20b-cloud"
EMBED_MODEL = "nomic-embed-text"
KNOWLEDGE_DIR = "./knowledge_base"
STORAGE_DIR = "./storage_graph"

WATCHER_DEBOUNCE_SECONDS = 5
WATCHER_VALID_EXTENSIONS = {".md", ".txt", ".pdf", ".docx"}

logging.basicConfig(level=logging.ERROR)


def _init_llm(model_name: str = MODEL_NAME, embed_model: str = EMBED_MODEL):
    """Initialize LLM and embedding model. Called from main() to avoid side effects on import."""
    Settings.llm = Ollama(
        model=model_name,
        request_timeout=300.0,
        base_url="http://localhost:11434",
    )
    Settings.embed_model = OllamaEmbedding(model_name=embed_model)

    # Chunking: large chunks to avoid splitting small notes
    Settings.chunk_size = 2048
    Settings.chunk_overlap = 256

# ==========================================
# SYSTEM PROMPT (multilingual)
# ==========================================
SYSTEM_PROMPT = (
    "You are the research assistant of Geode Graph. Your task is to answer questions "
    "based exclusively on the provided context from the user's knowledge base.\n\n"
    "RULES:\n"
    "1. Use ONLY information explicitly stated in the context below. Never invent or assume facts.\n"
    "2. Be concise but complete: include every relevant fact found in the context. "
    "Do not omit cited information that answers the question.\n"
    "3. If the answer involves multiple files, state which files are involved and "
    "quote the connecting fact from each.\n"
    "4. Always cite source file names in square brackets (e.g. [document.md]).\n"
    "5. When quoting actions or facts, preserve the original meaning. "
    "If a document says someone WILL do something, report it as a future action.\n"
    "6. When the user asks what to do BEFORE an event, include ALL tasks, preparations, "
    "and actions related to that event.\n"
    "7. If you cannot find the answer, say: 'I don't have enough information in your local files.'\n"
    "8. If unsure about a detail, omit it rather than guessing.\n"
    "9. ALWAYS respond in the same language as the user's question.\n\n"
    "CONTEXT:\n"
    "{context_str}\n\n"
    "QUESTION: {query_str}"
)

qa_template = PromptTemplate(SYSTEM_PROMPT)


# ==========================================
# SAFE PRINT (Rich-powered)
# ==========================================
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.status import Status

console = Console()


def safe_print(*args, **kwargs):
    try:
        console.print(*args, **kwargs)
    except Exception:
        try:
            print(*[str(a) for a in args])
        except Exception:
            pass


# ==========================================
# BM25 CHUNK RETRIEVER
# ==========================================
class BM25ChunkRetriever(CustomPGRetriever):
    """Multilingual BM25-like retriever that searches text chunks in the graph."""

    K1 = 1.5
    B = 0.75
    TOP_K = 8

    def init(self, **kwargs):
        self._idf_cache = {}
        self._avg_dl = 0
        self._doc_count = 0
        self._corpus_built = False

    def _build_corpus_stats(self, chunks: list[tuple[str, str]]):
        """Compute IDF and average document length."""
        if self._corpus_built:
            return

        self._doc_count = len(chunks)
        if self._doc_count == 0:
            return

        total_len = 0
        df = Counter()

        for _, text in chunks:
            tokens = tokenize(text)  # Usa tokenizer multilingue
            total_len += len(tokens)
            unique_tokens = set(tokens)
            for t in unique_tokens:
                df[t] += 1

        self._avg_dl = total_len / self._doc_count if self._doc_count > 0 else 1

        for token, freq in df.items():
            self._idf_cache[token] = math.log(
                1 + (self._doc_count - freq + 0.5) / (freq + 0.5)
            )

        self._corpus_built = True

    def _bm25_score(self, query_tokens: list[str], doc_text: str) -> float:
        """Compute BM25 score for a document."""
        doc_tokens = tokenize(doc_text)
        doc_len = len(doc_tokens)
        if doc_len == 0:
            return 0.0

        tf = Counter(doc_tokens)
        score = 0.0

        for qt in query_tokens:
            if qt not in self._idf_cache:
                continue
            term_freq = tf.get(qt, 0)
            if term_freq == 0:
                continue
            idf = self._idf_cache[qt]
            numerator = term_freq * (self.K1 + 1)
            denominator = term_freq + self.K1 * (
                1 - self.B + self.B * doc_len / self._avg_dl
            )
            score += idf * numerator / denominator

        return score

    def custom_retrieve(self, query_str: str) -> list[NodeWithScore]:
        """Search text chunks in the graph using BM25 scoring."""
        results = []
        chunks = []
        try:
            all_nodes = self.graph_store.graph.nodes
            for nid, data in all_nodes.items():
                text = getattr(data, "text", None)
                if text and len(text.strip()) > 20:
                    chunks.append((str(nid), text))
        except Exception:
            return results

        if not chunks:
            return results

        self._build_corpus_stats(chunks)

        query_tokens = tokenize(query_str)
        if not query_tokens:
            return results

        scored = []
        for nid, text in chunks:
            score = self._bm25_score(query_tokens, text)
            if score > 0:
                scored.append((nid, text, score))

        scored.sort(key=lambda x: x[2], reverse=True)
        for nid, text, score in scored[: self.TOP_K]:
            results.append(
                NodeWithScore(node=TextNode(text=text, id_=nid), score=score)
            )

        return results


# ==========================================
# TEMPORAL METADATA RETRIEVER
# ==========================================
class TemporalMetadataRetriever(CustomPGRetriever):
    """Multilingual retriever for temporal queries and events."""

    TOP_K = 8

    def init(self, **kwargs):
        pass

    def custom_retrieve(self, query_str: str) -> list[NodeWithScore]:
        """Search for relevant documents based on dates, tags and metadata in any language."""
        results = []

        query_date_tokens = extract_date_tokens(query_str)
        query_tokens = tokenize(query_str)

        is_temporal_query = bool(query_date_tokens) or bool(
            ALL_TEMPORAL_KEYWORDS.intersection(set(query_tokens))
        )

        if not is_temporal_query and not query_tokens:
            return results

        try:
            all_nodes = self.graph_store.graph.nodes
        except Exception:
            return results

        for nid, data in all_nodes.items():
            text = getattr(data, "text", None)
            if not text or len(text.strip()) < 20:
                continue

            score = 0.0
            text_lower = text.lower()

            # 1. Date matching
            for dt in query_date_tokens:
                if dt in text_lower:
                    score += 3.0

            # 2. Temporal keyword matching (all languages)
            for kw in ALL_TEMPORAL_KEYWORDS:
                if kw in query_tokens and kw in tokenize(text):
                    score += 2.0

            # 3. Tag/metadata matching
            if "tags:" in text_lower or "data:" in text_lower or "date:" in text_lower:
                for qt in query_tokens:
                    if qt in text_lower:
                        score += 1.0

            # 4. Proper name matching
            for qt in query_tokens:
                if len(qt) > 4 and qt in text_lower:
                    if qt[0:1].upper() + qt[1:] in text:
                        score += 1.5

            if score > 0:
                results.append(
                    NodeWithScore(node=TextNode(text=text, id_=str(nid)), score=score)
                )

        results.sort(key=lambda x: x.score, reverse=True)
        return results[: self.TOP_K]


# ==========================================
# READ-WRITE LOCK
# ==========================================
class ReadWriteLock:
    """Lock that allows concurrent reads but exclusive writes."""

    def __init__(self):
        self._read_ready = threading.Condition(threading.Lock())
        self._readers = 0

    def acquire_read(self):
        with self._read_ready:
            self._readers += 1

    def release_read(self):
        with self._read_ready:
            self._readers -= 1
            if self._readers == 0:
                self._read_ready.notify_all()

    def acquire_write(self):
        self._read_ready.acquire()
        while self._readers > 0:
            self._read_ready.wait()

    def release_write(self):
        self._read_ready.release()


# ==========================================
# OBSIDIAN PRE-PROCESSING (multilingual)
# ==========================================
_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Extract YAML frontmatter and return (metadata_dict, body_text).

    Handles BOM and leading whitespace that some editors add on Windows.
    """
    # Strip BOM and leading whitespace before matching
    text_clean = text.lstrip("\ufeff").lstrip()
    match = _FRONTMATTER_RE.match(text_clean)
    if not match:
        return {}, text
    try:
        meta = yaml.safe_load(match.group(1))
        if not isinstance(meta, dict):
            meta = {}
    except yaml.YAMLError:
        meta = {}
    # Calculate body offset relative to original text
    offset_in_clean = match.end()
    # Find where text_clean starts in original text
    prefix_len = len(text) - len(text.lstrip("\ufeff").lstrip())
    body = text[prefix_len + offset_in_clean:]
    return meta, body


def extract_wikilink_triples(file_path: str, text: str) -> list[tuple[str, str, str]]:
    """Extract structured triples from Obsidian [[wikilinks]] with multilingual inference."""
    filename = Path(file_path).stem
    triples = []
    seen = set()

    for match in _WIKILINK_RE.finditer(text):
        target = match.group(1).strip()
        if not target:
            continue

        pair_key = (filename.lower(), target.lower())
        if pair_key in seen:
            continue
        seen.add(pair_key)

        line_start = text.rfind("\n", 0, match.start()) + 1
        line_end = text.find("\n", match.end())
        if line_end == -1:
            line_end = len(text)
        line = text[line_start:line_end].strip()

        # Use multilingual inference
        relation = infer_relation(line, filename, target)
        triples.append((filename, relation, target))

    return triples


def extract_frontmatter_triples(
    file_path: str, metadata: dict
) -> list[tuple[str, str, str]]:
    """Generate structured triples from YAML frontmatter.

    Frontmatter keys are mapped to relations across supported languages.
    """
    filename = Path(file_path).stem
    triples = []

    # Frontmatter key -> relation mapping (multilingual)
    key_relations = {
        # Italian
        "ruolo": "Has role",
        "organizzazione": "Belongs to",
        "progetto": "Participates in",
        "stato": "Has status",
        "responsabile": "Has responsible",
        "licenza": "Has license",
        "location": "Located at",
        # English
        "role": "Has role",
        "organization": "Belongs to",
        "project": "Participates in",
        "status": "Has status",
        "responsible": "Has responsible",
        "license": "Has license",
        # French
        "organisation": "Belongs to",
        "projet": "Participates in",
        "statut": "Has status",
        "lieu": "Located at",
        # German
        "rolle": "Has role",
        "projekt": "Participates in",
        "standort": "Located at",
        # Spanish
        "rol": "Has role",
        "organizacion": "Belongs to",
        "proyecto": "Participates in",
        "estado": "Has status",
        "ubicacion": "Located at",
        # Portuguese
        "funcao": "Has role",
        "organizacao": "Belongs to",
        "projeto": "Participates in",
        "localizacao": "Located at",
    }

    for key, relation in key_relations.items():
        value = metadata.get(key)
        if value and isinstance(value, str):
            triples.append((filename, relation, value))

    # Tags (universal key)
    tags = metadata.get("tags", [])
    if isinstance(tags, list):
        for tag in tags:
            if isinstance(tag, str):
                triples.append((filename, "Has tag", tag))

    # Participants (multilingual keys)
    for key in ("partecipanti", "participants", "teilnehmer", "participantes"):
        partecipanti = metadata.get(key, [])
        if isinstance(partecipanti, list):
            for p in partecipanti:
                if isinstance(p, str):
                    triples.append((filename, "Has participant", p))

    # Date (multilingual keys)
    for key in ("data", "date", "datum", "fecha"):
        data = metadata.get(key)
        if data:
            triples.append((filename, "Has date", str(data)))
            break

    # Budget (universal)
    budget = metadata.get("budget")
    if budget:
        triples.append((filename, "Has budget", f"{budget}"))

    # Duration (multilingual keys)
    for key in ("durata_mesi", "duration_months", "duree_mois", "dauer_monate",
                "duracion_meses", "duracao_meses"):
        durata = metadata.get(key)
        if durata:
            triples.append((filename, "Has duration", f"{durata} months"))
            break

    return triples


def enrich_documents(
    documents: list[Document],
) -> tuple[list[Document], list[tuple[str, str, str]]]:
    """Pre-process documents: extract frontmatter, wikilinks and generate structured triples."""
    all_triples = []
    enriched_docs = []

    for doc in documents:
        text = doc.text
        file_path = doc.metadata.get("file_path", doc.id_)

        metadata, body = parse_frontmatter(text)

        for key, value in metadata.items():
            if isinstance(value, (str, int, float)):
                doc.metadata[f"fm_{key}"] = str(value)
            elif isinstance(value, list):
                doc.metadata[f"fm_{key}"] = ", ".join(str(v) for v in value)

        wikilink_triples = extract_wikilink_triples(file_path, body)
        all_triples.extend(wikilink_triples)

        fm_triples = extract_frontmatter_triples(file_path, metadata)
        all_triples.extend(fm_triples)

        enriched_docs.append(doc)

    all_triples = _deduplicate_triples(all_triples)
    return enriched_docs, all_triples


def _deduplicate_triples(
    triples: list[tuple[str, str, str]],
) -> list[tuple[str, str, str]]:
    """Remove duplicate triples (case-insensitive)."""
    seen = set()
    unique = []
    for s, r, o in triples:
        key = (s.lower().strip(), r.lower().strip(), o.lower().strip())
        if key not in seen:
            seen.add(key)
            unique.append((s.strip(), r.strip(), o.strip()))
    return unique


# ==========================================
# GRAPH RAG ENGINE
# ==========================================
class WritHerGraphRAG:
    def __init__(
        self,
        fast_mode: bool = False,
        model_name: str = MODEL_NAME,
        embed_model: str = EMBED_MODEL,
    ) -> None:
        self.index = None
        self._rw_lock = ReadWriteLock()
        self._query_engine = None
        self._retrievers_dirty = True
        self._fast_mode = fast_mode
        self.model_name = model_name
        self.embed_model = embed_model
        if not os.path.exists(KNOWLEDGE_DIR):
            os.makedirs(KNOWLEDGE_DIR)
        self.load_or_build_index()

    def load_or_build_index(self):
        """Load existing graph or build a new one."""
        self._rw_lock.acquire_write()
        try:
            try:
                if os.path.exists(STORAGE_DIR) and os.listdir(STORAGE_DIR):
                    # P1.4: Check embed model compatibility
                    self._check_storage_compatibility()
                    safe_print("Loading knowledge graph from local storage...")
                    storage_context = StorageContext.from_defaults(
                        persist_dir=STORAGE_DIR
                    )
                    self.index = load_index_from_storage(storage_context)
                    safe_print("Graph loaded successfully.")
                else:
                    self._build_index_unlocked()
            except Exception as e:
                safe_print(f"Load error: {e}. Rebuilding index...")
                self._build_index_unlocked()
            self._retrievers_dirty = True
        finally:
            self._rw_lock.release_write()

    def _check_storage_compatibility(self):
        """Verify that stored graph was built with the same embedding model.

        Raises RuntimeError if embed model mismatch is detected.
        """
        import json as _json

        meta_path = os.path.join(STORAGE_DIR, ".kwipu_meta.json")
        if not os.path.exists(meta_path):
            return  # Legacy storage without manifest, allow loading

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = _json.load(f)
        except (OSError, _json.JSONDecodeError):
            return

        stored_embed = meta.get("embed_model", "")
        current_embed = self.embed_model

        if stored_embed and stored_embed != current_embed:
            raise RuntimeError(
                f"Embedding model mismatch: storage was built with '{stored_embed}' "
                f"but current config uses '{current_embed}'. "
                f"Delete '{STORAGE_DIR}/' to rebuild, or restore the previous model."
            )

        # LLM model change is fine (only used for generation, not embeddings)
        stored_llm = meta.get("llm_model", "")
        if stored_llm and stored_llm != self.model_name:
            safe_print(
                f"[dim]Note: graph was built with '{stored_llm}', "
                f"now using '{self.model_name}' for queries.[/dim]"
            )

    def _save_storage_manifest(self):
        """Save metadata about the current build configuration."""
        import json as _json

        meta_path = os.path.join(STORAGE_DIR, ".kwipu_meta.json")
        meta = {
            "embed_model": self.embed_model,
            "llm_model": self.model_name,
            "version": "1.0",
        }
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                _json.dump(meta, f, indent=2)
        except OSError:
            pass

    def build_index(self):
        """Rebuild the graph (thread-safe with write lock)."""
        self._rw_lock.acquire_write()
        try:
            self._build_index_unlocked()
            self._retrievers_dirty = True
        finally:
            self._rw_lock.release_write()

    def insert_document(self, file_path):
        """Insert a single document into the existing graph (incremental)."""
        self._rw_lock.acquire_write()
        try:
            if not self.index:
                self._build_index_unlocked()
                self._retrievers_dirty = True
                return

            try:
                reader = SimpleDirectoryReader(
                    input_files=[file_path], filename_as_id=True
                )
                docs = reader.load_data()
                if docs:
                    enriched_docs, structural_triples = enrich_documents(docs)
                    for doc in enriched_docs:
                        safe_print(
                            f"Incremental insert: {os.path.basename(file_path)}..."
                        )
                        self.index.insert(doc)
                    self._inject_structural_triples(structural_triples)
                    self.index.storage_context.persist(persist_dir=STORAGE_DIR)
                    safe_print("Document added to graph successfully.")
                    self._retrievers_dirty = True
            except Exception as e:
                safe_print(
                    f"Incremental insert error: {e}. Full rebuild..."
                )
                self._build_index_unlocked()
                self._retrievers_dirty = True
        finally:
            self._rw_lock.release_write()

    def update_document(self, file_path):
        """Update a modified document in the graph (delete + re-insert).

        More efficient than a full rebuild for single-file modifications.
        Falls back to full rebuild if the incremental update fails.
        """
        self._rw_lock.acquire_write()
        try:
            if not self.index:
                self._build_index_unlocked()
                self._retrievers_dirty = True
                return

            try:
                # The ref_doc_id is the file path (set by filename_as_id=True)
                ref_doc_id = file_path
                safe_print(f"Updating: {os.path.basename(file_path)}...")

                # Step 1: Remove old version from graph
                self.index.delete_ref_doc(ref_doc_id, delete_from_docstore=True)

                # Step 2: Re-read and insert updated version
                reader = SimpleDirectoryReader(
                    input_files=[file_path], filename_as_id=True
                )
                docs = reader.load_data()
                if docs:
                    enriched_docs, structural_triples = enrich_documents(docs)
                    for doc in enriched_docs:
                        self.index.insert(doc)
                    self._inject_structural_triples(structural_triples)

                self.index.storage_context.persist(persist_dir=STORAGE_DIR)
                safe_print("Document updated successfully.")
                self._retrievers_dirty = True
            except Exception as e:
                safe_print(
                    f"Incremental update error: {e}. Full rebuild..."
                )
                self._build_index_unlocked()
                self._retrievers_dirty = True
        finally:
            self._rw_lock.release_write()

    def _inject_structural_triples(self, triples: list[tuple[str, str, str]]):
        """Inject pre-extracted triples into the property graph."""
        if not self.index or not triples:
            return
        graph_store = self.index.property_graph_store
        for subj, rel, obj in triples:
            try:
                graph_store.upsert_triplet(subj, rel, obj)
            except Exception:
                pass

    def _build_index_unlocked(self):
        """Analyze files and build the graph."""
        safe_print(f"Scanning documents in '{KNOWLEDGE_DIR}'...")

        try:
            reader = SimpleDirectoryReader(
                KNOWLEDGE_DIR, recursive=True, filename_as_id=True
            )
            documents = reader.load_data()
        except ValueError:
            documents = []

        if not documents:
            safe_print("No files found. Waiting for documents...")
            self.index = None
            return

        # Time estimate for user feedback
        n_docs = len(documents)
        safe_print(f"Found {n_docs} documents.")
        if n_docs > 10:
            est_minutes = max(1, n_docs // 3)
            safe_print(
                f"  ⏱  Estimate: {est_minutes}-{est_minutes * 3} minutes "
                f"(depends on model and hardware)."
            )
            safe_print(
                "  💡 Tip: first build is the slowest. "
                "Subsequent runs will be incremental."
            )

        safe_print("Pre-processing: extracting wikilinks and frontmatter...")
        enriched_docs, structural_triples = enrich_documents(documents)
        safe_print(
            f"  -> {len(structural_triples)} structural relations extracted."
        )

        build_start = time.time()
        safe_print(
            f"LLM extraction and graph construction with {self.model_name}..."
        )

        def parse_triplets(response_str, max_length=128):
            results = []
            for line in response_str.strip().split("\n"):
                line = line.strip().strip("-").strip("*").strip()
                if not line:
                    continue
                if "(" in line and ")" in line:
                    line = line[line.index("(") + 1 : line.index(")")]
                tokens = line.split(",")
                if len(tokens) != 3:
                    continue
                subj, pred, obj = (t.strip().strip('"') for t in tokens)
                if not subj or not pred or not obj:
                    continue
                if any(len(s.encode("utf-8")) > max_length for s in [subj, pred, obj]):
                    continue
                results.append(
                    (subj.capitalize(), pred.capitalize(), obj.capitalize())
                )
            return results

        kg_extractors = [
            SimpleLLMPathExtractor(
                llm=Settings.llm,
                extract_prompt=(
                    "From the text below, extract up to {max_paths_per_chunk} knowledge triplets.\n"
                    "Each triplet must be on its own line in the format: entity1, relation, entity2\n"
                    "RULES:\n"
                    "- Each entity must be a single proper noun (person, organization, place, technology, dataset)\n"
                    "- Do NOT combine multiple concepts into one entity\n"
                    "- Extract ALL person names as separate entities\n"
                    "- Relations should be short verb phrases\n\n"
                    "Text: {text}\n"
                    "Triplets:\n"
                ),
                parse_fn=parse_triplets,
                num_workers=1,
                max_paths_per_chunk=20,
            ),
            ImplicitPathExtractor(),
        ]

        self.index = PropertyGraphIndex.from_documents(
            enriched_docs, kg_extractors=kg_extractors, show_progress=False
        )

        safe_print("Injecting structural relations into graph...")
        self._inject_structural_triples(structural_triples)

        self.index.storage_context.persist(persist_dir=STORAGE_DIR)
        self._save_storage_manifest()
        build_elapsed = time.time() - build_start
        minutes = int(build_elapsed // 60)
        seconds = int(build_elapsed % 60)
        safe_print(
            f"Graph built and saved successfully. "
            f"(Build time: {minutes}m {seconds}s)"
        )

    def _build_retrievers(self):
        """Build retrievers and query engine.

        Fast mode: vector + BM25 + temporal only (no LLM call per query).
        Normal mode: adds LLM synonym retriever.
        """
        if not self.index:
            self._query_engine = None
            return

        sub_retrievers = []

        # Synonym retriever: normal mode only (costs one LLM call per query)
        if not self._fast_mode:
            synonym_retriever = LLMSynonymRetriever(
                self.index.property_graph_store,
                llm=Settings.llm,
                include_text=True,
                synonym_prompt=(
                    "Given the query below, generate synonyms or related keywords up to {max_keywords} in total.\n"
                    "Include: original names, names with titles (Prof., Dott., Dott.ssa, Dr., Ing., Dra.), "
                    "abbreviations, related project names, and multilingual variants.\n"
                    "Provide all synonyms/keywords separated by '^' symbols: 'keyword1^keyword2^...'\n"
                    "Result must be one line, separated by '^' symbols.\n"
                    "----\n"
                    "QUERY: {query_str}\n"
                    "----\n"
                    "KEYWORDS: "
                ),
                max_keywords=15,
                path_depth=3,
            )
            sub_retrievers.append(synonym_retriever)

        # These retrievers don't use the LLM, always active
        vector_retriever = VectorContextRetriever(
            self.index.property_graph_store,
            vector_store=self.index.vector_store,
            include_text=True,
            similarity_top_k=20,
            embed_model=Settings.embed_model,
            path_depth=3,
        )
        sub_retrievers.append(vector_retriever)

        bm25_retriever = BM25ChunkRetriever(self.index.property_graph_store)
        sub_retrievers.append(bm25_retriever)

        temporal_retriever = TemporalMetadataRetriever(self.index.property_graph_store)
        sub_retrievers.append(temporal_retriever)

        self._query_engine = self.index.as_query_engine(
            text_qa_template=qa_template,
            sub_retrievers=sub_retrievers,
        )
        self._retrievers_dirty = False

    def ask(self, question):
        """Query the graph with read lock.

        Uses a lock-upgrade pattern: acquire read to check state, release,
        then acquire write only if retrievers need rebuilding, then read again for query.
        """
        self._rw_lock.acquire_read()
        try:
            if not self.index:
                return "No index available. Add files to the knowledge_base folder."
            needs_rebuild = self._retrievers_dirty
        finally:
            self._rw_lock.release_read()

        if needs_rebuild:
            self._rw_lock.acquire_write()
            try:
                if self._retrievers_dirty:
                    self._build_retrievers()
            finally:
                self._rw_lock.release_write()

        self._rw_lock.acquire_read()
        try:
            if not self._query_engine:
                return "No index available. Add files to the knowledge_base folder."
            response = self._query_engine.query(question)
            return response
        finally:
            self._rw_lock.release_read()


# ==========================================
# REAL-TIME FILE MONITORING (with persistent content-hash)
# ==========================================
_HASH_CACHE_FILE = os.path.join(STORAGE_DIR, ".file_hashes.json")


def _file_content_hash(path: str) -> str | None:
    """Compute MD5 hash of file contents. Returns None if file doesn't exist."""
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except (OSError, IOError):
        return None


def _load_hash_cache() -> dict[str, str]:
    """Load hash cache from disk."""
    import json

    try:
        if os.path.exists(_HASH_CACHE_FILE):
            with open(_HASH_CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def _save_hash_cache(hashes: dict[str, str]):
    """Save hash cache to disk."""
    import json

    os.makedirs(os.path.dirname(_HASH_CACHE_FILE), exist_ok=True)
    try:
        with open(_HASH_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(hashes, f, indent=2)
    except OSError:
        pass


class FileWatcher(FileSystemEventHandler):
    def __init__(self, rag_system):
        self.rag_system = rag_system
        self._lock = threading.Lock()
        self._pending_events: dict[str, tuple[str, float]] = {}
        self._timer = None
        # Persistent hash cache on disk
        self._file_hashes: dict[str, str] = _load_hash_cache()
        self._refresh_hashes()

    def _refresh_hashes(self):
        """Refresh hashes for all current files in the knowledge base."""
        kb_path = Path(KNOWLEDGE_DIR)
        if not kb_path.exists():
            return

        current_files = set()
        for ext in WATCHER_VALID_EXTENSIONS:
            for f in kb_path.rglob(f"*{ext}"):
                if ".obsidian" not in f.parts:
                    fpath = str(f)
                    current_files.add(fpath)
                    h = _file_content_hash(fpath)
                    if h and fpath not in self._file_hashes:
                        self._file_hashes[fpath] = h

        # Remove hashes for deleted files
        stale = [k for k in self._file_hashes if k not in current_files]
        for k in stale:
            del self._file_hashes[k]

        _save_hash_cache(self._file_hashes)

    def _is_relevant_file(self, path):
        p = Path(path)
        if ".obsidian" in p.parts:
            return False
        return p.suffix.lower() in WATCHER_VALID_EXTENSIONS

    def _has_content_changed(self, path: str) -> bool:
        """Check if file content has actually changed compared to last hash."""
        new_hash = _file_content_hash(path)
        if new_hash is None:
            return path in self._file_hashes

        old_hash = self._file_hashes.get(path)
        if old_hash == new_hash:
            return False  # Identical content, phantom event

        self._file_hashes[path] = new_hash
        _save_hash_cache(self._file_hashes)
        return True

    def _schedule_processing(self, event_type, path):
        with self._lock:
            self._pending_events[path] = (event_type, time.time())
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(
                WATCHER_DEBOUNCE_SECONDS, self._process_pending
            )
            self._timer.daemon = True
            self._timer.start()

    def _process_pending(self):
        with self._lock:
            events = dict(self._pending_events)
            self._pending_events.clear()
            self._timer = None

        if not events:
            return

        # Filter phantom events: verify content actually changed
        real_events = {}
        for path, (etype, ts) in events.items():
            if etype == "deleted":
                # Deletions are always real
                self._file_hashes.pop(path, None)
                real_events[path] = (etype, ts)
            elif self._has_content_changed(path):
                real_events[path] = (etype, ts)

        if not real_events:
            safe_print("\n(Filesystem events ignored: no real content changes)")
            return

        # Separate events by type
        deleted = [p for p, (e, _) in real_events.items() if e == "deleted"]
        modified = [p for p, (e, _) in real_events.items() if e == "modified"]
        created = [p for p, (e, _) in real_events.items() if e == "created"]

        # Deletions require full rebuild (can't selectively remove all related triples)
        if deleted:
            safe_print(f"\nFile(s) deleted. Rebuilding graph...")
            self.rag_system.build_index()
            return

        # Modifications: incremental update (delete + re-insert per file)
        for path in modified:
            if os.path.exists(path):
                self.rag_system.update_document(path)

        # Creations: incremental insert
        for path in created:
            if os.path.exists(path):
                safe_print(f"\nNew file detected: {os.path.basename(path)}.")
                self.rag_system.insert_document(path)

    def on_created(self, event):
        if not event.is_directory and self._is_relevant_file(event.src_path):
            self._schedule_processing("created", event.src_path)

    def on_modified(self, event):
        if not event.is_directory and self._is_relevant_file(event.src_path):
            self._schedule_processing("modified", event.src_path)

    def on_deleted(self, event):
        if not event.is_directory and self._is_relevant_file(event.src_path):
            self._schedule_processing("deleted", event.src_path)


# ==========================================
# TERMINAL INTERFACE (Rich)
# ==========================================
def _check_ollama_available(model_name: str, embed_model: str):
    """Verify Ollama is running and required models are available.

    Prints clear error messages with suggested commands if something is missing.
    Returns True if everything is ready, False otherwise.
    """
    import urllib.request
    import json as _json

    base_url = "http://localhost:11434"

    # Check if Ollama is running
    try:
        req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = _json.loads(resp.read().decode())
    except Exception:
        console.print(
            Panel(
                "[bold red]Ollama is not running.[/bold red]\n\n"
                "Start Ollama before running Geode Graph:\n"
                "  [dim]ollama serve[/dim]",
                title="[red]Connection Error[/red]",
                border_style="red",
            )
        )
        return False

    # Check available models
    available_models = set()
    for model_info in data.get("models", []):
        name = model_info.get("name", "")
        available_models.add(name)
        # Also add without tag (e.g. "qwen2.5:3b" -> "qwen2.5")
        if ":" in name:
            available_models.add(name.split(":")[0])

    missing = []
    if model_name not in available_models and model_name.split(":")[0] not in available_models:
        missing.append(model_name)
    if embed_model not in available_models and embed_model.split(":")[0] not in available_models:
        missing.append(embed_model)

    if missing:
        cmds = "\n".join(f"  [dim]ollama pull {m}[/dim]" for m in missing)
        console.print(
            Panel(
                f"[bold yellow]Missing model(s):[/bold yellow] {', '.join(missing)}\n\n"
                f"Pull them with:\n{cmds}",
                title="[yellow]Model Not Found[/yellow]",
                border_style="yellow",
            )
        )
        return False

    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Geode Graph - Knowledge Graph Assistant")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: disables LLM synonym retriever for faster queries",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help=f"Override LLM model (default: {MODEL_NAME})",
    )
    parser.add_argument(
        "--embed-model",
        type=str,
        default=None,
        help=f"Override embedding model (default: {EMBED_MODEL})",
    )
    args = parser.parse_args()

    # Resolve model names (CLI overrides config)
    llm_model = args.llm_model or MODEL_NAME
    embed_model = args.embed_model or EMBED_MODEL

    _ensure_nest_asyncio()

    # Check Ollama before doing anything expensive
    if not _check_ollama_available(llm_model, embed_model):
        sys.exit(1)

    # Initialize LLM (P0.6: no side effects on import)
    _init_llm(model_name=llm_model, embed_model=embed_model)

    console.print()
    console.print(
        Panel(
            Text.from_markup(
                f"[bold]Geode Graph[/bold]\n"
                f"[dim]LLM:[/dim] {llm_model}  "
                f"[dim]Mode:[/dim] {'FAST' if args.fast else 'FULL'}  "
                f"[dim]Watching:[/dim] {KNOWLEDGE_DIR}"
            ),
            border_style="bright_black",
            padding=(1, 2),
        )
    )

    with Status("[dim]Loading knowledge graph...[/dim]", console=console, spinner="dots"):
        rag = WritHerGraphRAG(
            fast_mode=args.fast,
            model_name=llm_model,
            embed_model=embed_model,
        )

    observer = Observer()
    observer.schedule(FileWatcher(rag), KNOWLEDGE_DIR, recursive=True)
    observer.start()

    console.print("[dim]Type your question, or 'exit' to quit.[/dim]\n")

    try:
        while True:
            try:
                query = console.input("[bold bright_white]>[/bold bright_white] ")
            except EOFError:
                break

            if query.lower().strip() in ["exit", "quit", "esci"]:
                break

            if not query.strip():
                continue

            with Status("[dim]Querying graph...[/dim]", console=console, spinner="dots"):
                start_t = time.time()
                response = rag.ask(query)
                elapsed = time.time() - start_t

            console.print()
            console.print(
                Panel(
                    Markdown(str(response)),
                    title="[bold]Response[/bold]",
                    subtitle=f"[dim]{elapsed:.1f}s[/dim]",
                    border_style="bright_black",
                    padding=(1, 2),
                )
            )
            console.print()

    except KeyboardInterrupt:
        pass
    finally:
        observer.stop()
        observer.join()
        console.print("\n[dim]Goodbye.[/dim]")


if __name__ == "__main__":
    main()
