"""FastAPI integration for PyPond pipelines.

This module provides FastAPI integration that allows PyPond pipelines to be
executed via HTTP endpoints while maintaining the same pipeline definitions.
Supports automatic URL derivation from PyPond paths and proper handling of
both regular pydantic fields and File[DataT] fields.
"""
