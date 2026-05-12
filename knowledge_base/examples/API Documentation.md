---
tags: [resource, documentation, api]
status: published
owner: Bob
---

# API Documentation

Internal API docs for [[Project Alpha]], maintained by [[Bob]].

## Endpoints
- `POST /onboarding/start` - Initiate customer onboarding flow
- `GET /onboarding/status/:id` - Check onboarding progress
- `PUT /onboarding/complete/:id` - Mark onboarding as complete

## Authentication
- OAuth2 with JWT tokens
- Tokens expire after 24 hours

## Notes
- [[Alice]] uses these endpoints for the frontend integration
- [[Charlie]] requested rate limiting after the load test in January
- Docs hosted on internal wiki, link shared in [[Meeting Notes - Jan 15]]
