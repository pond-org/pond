# Copyright 2025 Nils Bore
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
