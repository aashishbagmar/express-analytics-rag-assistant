# Request Bodies and Validation

FastAPI validates request bodies by combining Pydantic models with standard
Python type hints. When a path operation declares a parameter whose type is
a Pydantic `BaseModel` subclass, FastAPI knows the data should be read from
the request body, parsed as JSON, and validated against that model's field
types before your endpoint function ever runs.

## Declaring a request body model

```python
from fastapi import FastAPI
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    price: float
    is_offer: bool | None = None

app = FastAPI()

@app.post("/items/")
async def create_item(item: Item):
    return item
```

When a client sends a POST request to `/items/`, FastAPI:

1. Reads the raw request body as JSON.
2. Validates each field against the `Item` model's declared types
   (`str`, `float`, `bool | None`), applying Pydantic's type coercion rules
   where sensible (for example, converting a numeric string to `float`).
3. If any field is missing, has the wrong type, or fails a validation
   constraint, FastAPI automatically returns an HTTP `422 Unprocessable
   Entity` response with a structured JSON body describing exactly which
   fields failed and why - your endpoint code never even runs.
4. If validation succeeds, FastAPI passes a fully-typed `Item` instance into
   your function as the `item` parameter, so your code can immediately use
   dot-notation attribute access (`item.name`, `item.price`) with full
   editor autocomplete and type-checking support.

## Nested and complex validation

Pydantic models can be nested inside one another, and FastAPI validates the
entire nested structure recursively:

```python
class Address(BaseModel):
    street: str
    city: str

class Customer(BaseModel):
    name: str
    address: Address
```

Fields can also declare validation constraints directly using Pydantic's
`Field()`, for example enforcing minimum/maximum length or numeric ranges,
and FastAPI enforces these automatically as part of the same request-body
validation step, without any extra code in the endpoint itself.

## Multiple body parameters

A path operation can declare more than one Pydantic model parameter; FastAPI
then expects the request body to be a JSON object with one key per
parameter name, and validates each nested object against its corresponding
model independently.
