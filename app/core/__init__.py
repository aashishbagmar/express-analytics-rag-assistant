"""
Core utilities package.

Purpose:
    Hold cross-cutting configuration and logging boundaries.

Responsibilities:
    - Keep environment/configuration loading centralized.
    - Keep logging setup consistent across API, graph, ingestion, and scripts.

Public interfaces:
    - config.Settings
    - logging.configure_logging
"""
