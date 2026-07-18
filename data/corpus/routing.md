# Routing and Path Operations

In FastAPI, a "path operation" is the combination of an HTTP method (GET,
POST, PUT, DELETE, etc.) and a URL path, declared using decorators on your
`FastAPI` app instance or on an `APIRouter`.

## Basic path operations

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}
```

The `{item_id}` segment is a path parameter. Because the function parameter
`item_id` is type-hinted as `int`, FastAPI automatically converts the string
from the URL into an `int`, and returns a `422` validation error if the
value cannot be converted (for example, if a client requests `/items/abc`).

## Query parameters

Any function parameter that is not part of the path and is not a Pydantic
model is treated as a query parameter by default:

```python
@app.get("/items/")
async def list_items(skip: int = 0, limit: int = 10):
    return {"skip": skip, "limit": limit}
```

Here, `skip` and `limit` are optional query parameters (`?skip=0&limit=10`)
with default values, so a request to `/items/` without any query string
still works.

## Organizing routes with APIRouter

For larger applications, related path operations are usually grouped into an
`APIRouter` in their own module, then included in the main app:

```python
from fastapi import APIRouter

router = APIRouter(prefix="/users", tags=["users"])

@router.get("/")
async def list_users():
    return []

# in main.py
app.include_router(router)
```

This keeps routing organized by feature/domain and allows a shared prefix
and OpenAPI tag to be applied to every route registered on that router,
rather than repeating them on every single decorator.

## Route ordering

FastAPI matches routes in the order they are declared. A more specific path
(such as `/users/me`) must be declared before a more general path parameter
route (such as `/users/{user_id}`), or the general route will match first
and the specific one will never be reached.
