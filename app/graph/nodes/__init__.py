"""
LangGraph node package.

Purpose:
    Group node boundaries for the RAG workflow.

Responsibilities:
    - Keep each graph step in a small, named module.
    - Delegate real work to services rather than embedding business logic here.
    - Make node order easy for reviewers to inspect.

Public interfaces:
    Node modules expose one callable stub per graph node.
"""
