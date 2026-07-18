# Dependency Injection in FastAPI

FastAPI has a powerful but simple dependency injection system that lets you
share reusable logic - such as database connections, authentication checks,
or shared query parameters - across many path operations without duplicating
code.

## Defining a dependency

A dependency is just a function (or callable class) that FastAPI calls for
you before running your path operation function. You declare a dependency
using `Depends()`:

```python
from fastapi import Depends, FastAPI

app = FastAPI()

def get_query_params(q: str | None = None, limit: int = 10):
    return {"q": q, "limit": limit}

@app.get("/items/")
async def read_items(params: dict = Depends(get_query_params)):
    return params
```

When a request comes in, FastAPI first calls `get_query_params`, resolving
its own parameters (`q`, `limit`) from the incoming request just like it
would for a normal path operation, and then passes the dependency's return
value into `read_items` as the `params` argument.

## Why use dependencies

- **Reuse**: the same dependency function can be shared across dozens of
  endpoints (for example, "get the current authenticated user") without
  copy-pasting logic.
- **Testability**: dependencies can be overridden in tests via
  `app.dependency_overrides`, letting you substitute a fake database or fake
  user without touching endpoint code.
- **Composability**: dependencies can themselves depend on other
  dependencies, forming a dependency tree that FastAPI resolves
  automatically and efficiently, caching a dependency's result within a
  single request by default so it is not computed twice.

## Class-based dependencies

Dependencies do not have to be plain functions. Any callable, including a
class, can be used with `Depends()`:

```python
class Pagination:
    def __init__(self, skip: int = 0, limit: int = 10):
        self.skip = skip
        self.limit = limit

@app.get("/users/")
async def list_users(pagination: Pagination = Depends(Pagination)):
    return {"skip": pagination.skip, "limit": pagination.limit}
```

## Dependencies with yield

A dependency can use `yield` instead of `return` to run cleanup code after
the response has been sent - a common pattern for opening and closing a
database session around a request:

```python
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```
