"""Shared utilities for FastAPI transforms."""

import threading

import uvicorn
from fastapi import FastAPI

from pond.state import State
from pond.transforms.abstract_transform import AbstractExecuteUnit


class BaseExecuteUnit(AbstractExecuteUnit):
    """Base execute unit for FastAPI transforms with common functionality."""

    def __init__(self, inputs: list[str], outputs: list[str]):
        """Initialize the base FastAPI execute unit.

        Args:
            inputs: List of input path strings for this unit.
            outputs: List of output path strings for this unit.
        """
        super().__init__(inputs=inputs, outputs=outputs)
        self.state_ref: State | None = None

    def start_server(self, app: FastAPI, title: str) -> None:
        """Start the FastAPI server with the given app.

        Args:
            app: FastAPI application instance to run.
            title: Server title for logging.
        """
        print(f"{title} server started on http://{self.host}:{self.port}")
        uvicorn.run(app, host=self.host, port=self.port, log_level="info")

    def start_server_threaded(self, app: FastAPI, title: str) -> threading.Thread:
        """Start the FastAPI server in a background thread.

        Args:
            app: FastAPI application instance to run.
            title: Server title for logging.

        Returns:
            Thread object running the server.
        """

        def run_server():
            self.start_server(app, title)

        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        return server_thread
