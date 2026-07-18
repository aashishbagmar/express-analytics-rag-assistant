# Introduction to FastAPI

FastAPI is a modern, high-performance Python web framework for building APIs.
It is built on top of Starlette for the web parts and Pydantic for the data
parts, and it is designed around standard Python type hints.

## Why FastAPI

FastAPI was created to combine the best ideas from previous frameworks: fast
to run, fast to code, fewer bugs, and easy to learn. Because endpoint
parameters and request/response bodies are declared using standard Python
type hints, FastAPI can automatically:

- Validate incoming request data and return clear, structured errors when it
  does not match the declared types.
- Serialize response data to JSON, converting Python objects (including
  Pydantic models) automatically.
- Generate interactive API documentation (Swagger UI and ReDoc) directly from
  your code, with zero extra configuration.

## Creating an application

A minimal FastAPI application looks like this:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Hello, world"}
```

Running `uvicorn main:app --reload` starts a local development server with
hot reloading. Visiting `/docs` shows the automatically generated interactive
API documentation.

## Async support

FastAPI supports both `async def` and regular `def` path operation functions.
Using `async def` allows FastAPI to run the endpoint concurrently with other
requests when the function performs I/O-bound work (such as calling a
database or an external API) using `await`. Regular `def` functions are run
in a thread pool so that blocking code does not block the entire event loop.
