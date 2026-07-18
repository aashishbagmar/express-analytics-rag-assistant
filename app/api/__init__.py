"""
API router package.

Purpose:
    Group FastAPI route modules by endpoint area.

Responsibilities:
    - Keep request/response handling thin.
    - Delegate orchestration to the graph and services layers.
    - Avoid business logic inside route modules.

Public interfaces:
    Route modules expose router factory stubs until endpoint logic is added.
"""
