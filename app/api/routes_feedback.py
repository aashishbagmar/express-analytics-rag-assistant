"""
Feedback endpoint skeleton.

Purpose:
    Define the API boundary for collecting thumbs-up/down feedback on answers.

Responsibilities:
    - Accept validated feedback requests.
    - Validate query_id against the query registry/log.
    - Delegate persistence to the feedback store.

Public interfaces:
    - create_router: returns the feedback router when implemented.
"""


def create_router():
    """Create the feedback API router."""
