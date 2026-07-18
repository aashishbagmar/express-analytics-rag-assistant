# Response Models

FastAPI lets you declare the shape of a path operation's response using the
`response_model` parameter, which is validated and filtered the same way
request bodies are, but in the opposite direction.

## Declaring a response model

```python
from fastapi import FastAPI
from pydantic import BaseModel

class UserOut(BaseModel):
    username: str
    email: str

class UserIn(BaseModel):
    username: str
    email: str
    password: str

app = FastAPI()

@app.post("/users/", response_model=UserOut)
async def create_user(user: UserIn):
    return user
```

Even though the endpoint function returns the full `UserIn` object
(including the `password` field), FastAPI filters the outgoing response
through `UserOut`, so the field not declared on the response model - the
password - is automatically stripped out before the response is ever sent to
the client. This makes `response_model` a reliable way to prevent
accidentally leaking sensitive fields.

## Why declare response_model instead of relying on the return type hint

While FastAPI can infer some response behavior from a function's return type
annotation, explicitly declaring `response_model` gives you finer control:

- It documents the exact response shape in the generated OpenAPI schema,
  independent of any internal object your endpoint happens to return.
- It lets you return an ORM object or dict internally while still enforcing
  a clean, filtered public API contract.
- It supports `response_model_exclude_unset=True` to omit fields that were
  never explicitly set, useful for partial-update endpoints.

## Status codes and response model unions

`response_model` can be combined with `status_code` to declare different
shapes for different outcomes, and Pydantic's `Union` types let a single
endpoint declare more than one possible successful response shape, with
FastAPI validating the actual returned object against whichever member of
the union it matches.
