"""FastAPI hook for real-time pipeline execution updates and control."""

import asyncio
import json
import queue
import threading
from typing import List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from pond.hooks.abstract_hook import AbstractHook
from pond.transforms.abstract_transform import AbstractExecuteTransform
from pond.transforms.transform_pipe import TransformPipe


class FastAPIHook(AbstractHook):
    """Unified FastAPI hook for pipeline progress tracking and control.

    This hook registers both progress monitoring and control endpoints on a FastAPI app.
    It broadcasts real-time pipeline execution progress to connected clients and provides
    REST endpoints for pipeline control operations (reset, cancel, clear).

    Progress Endpoints:
    - WebSocket: `/progress/ws` for bidirectional real-time updates
    - SSE: `/progress/sse` for unidirectional streaming updates
    - GET: `/progress/status` for current progress status

    Control Endpoints:
    - POST: `/pipeline/reset` - Reset pipeline state
    - POST: `/pipeline/cancel` - Cancel current execution
    - DELETE: `/pipeline/clear` - Clear output data

    Attributes:
        app: FastAPI application instance to register endpoints on.
        connections: List of active WebSocket connections for broadcasting.
        current_status: Current pipeline execution status for new connections.
        cancel_requested: Thread-safe flag indicating if cancellation was requested.
        reset_requested: Thread-safe flag indicating if reset was requested.

    Note:
        This hook provides both real-time progress updates and pipeline control
        in a single unified interface. Supports cancellation checking for runners.
    """

    def __init__(self, app: FastAPI):
        """Initialize the unified FastAPI hook.

        Args:
            app: FastAPI application instance to register endpoints on.
        """
        super().__init__()
        self.app = app
        self.connections: List[WebSocket] = []
        self.current_status = {
            "stage": "idle",
            "total_transforms": 0,
            "completed_transforms": 0,
            "current_transform": None,
            "success": True,
            "error": None,
            "progress_percentage": 0,
        }
        # Control state
        self._lock = threading.Lock()
        self._cancel_requested = threading.Event()
        self._reset_requested = threading.Event()

        # Queue for thread-safe broadcasting
        self._broadcast_queue = queue.Queue()

        self._setup_endpoints()

    def _setup_endpoints(self):
        """Register WebSocket and SSE endpoints on the FastAPI app."""

        @self.app.websocket("/progress/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time progress updates."""
            await websocket.accept()
            self.connections.append(websocket)

            # Send current status to new connection
            await websocket.send_text(json.dumps(self.current_status))

            try:
                while True:
                    # Check for queued updates with timeout
                    try:
                        # Check for updates from the pipeline thread
                        while True:
                            try:
                                update = self._broadcast_queue.get_nowait()
                                await websocket.send_text(json.dumps(update))
                            except queue.Empty:
                                break
                    except Exception:
                        # Error sending update, connection might be closed
                        break

                    # Wait a bit before checking again, or handle client messages
                    try:
                        await asyncio.wait_for(websocket.receive_text(), timeout=0.1)
                    except asyncio.TimeoutError:
                        # No client message, continue checking for updates
                        continue
                    except WebSocketDisconnect:
                        break
            except WebSocketDisconnect:
                pass
            finally:
                if websocket in self.connections:
                    self.connections.remove(websocket)

        @self.app.get("/progress/sse")
        async def sse_endpoint():
            """Server-Sent Events endpoint for streaming progress updates."""

            async def event_generator():
                # Send current status first
                yield f"data: {json.dumps(self.current_status)}\n\n"

                # Keep connection alive and send updates
                last_status = self.current_status.copy()
                while True:
                    await asyncio.sleep(0.1)  # Check for updates every 100ms

                    # Send update if status changed
                    if self.current_status != last_status:
                        yield f"data: {json.dumps(self.current_status)}\n\n"
                        last_status = self.current_status.copy()

                        # End stream if pipeline completed or failed
                        if self.current_status["stage"] in ["completed", "failed"]:
                            break

            return StreamingResponse(
                event_generator(),
                media_type="text/plain",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Content-Type": "text/event-stream",
                },
            )

        @self.app.get("/progress/status")
        async def status_endpoint():
            """REST endpoint for current progress status."""
            return self.current_status

        # Control endpoints
        @self.app.post("/pipeline/reset")
        async def reset_pipeline():
            """Reset pipeline state and restart input collection."""
            with self._lock:
                try:
                    # Reset progress state
                    self.current_status = {
                        "stage": "idle",
                        "total_transforms": 0,
                        "completed_transforms": 0,
                        "current_transform": None,
                        "success": True,
                        "error": None,
                        "progress_percentage": 0,
                    }

                    # Clear control flags
                    self._cancel_requested.clear()
                    self._reset_requested.set()  # Signal reset was requested

                    # Broadcast reset to connected clients
                    await self._broadcast_update()

                    return {
                        "status": "reset_complete",
                        "message": "Pipeline state has been reset",
                        "stage": "idle",
                    }

                except Exception as e:
                    raise HTTPException(
                        status_code=500, detail=f"Reset failed: {str(e)}"
                    )

        @self.app.post("/pipeline/cancel")
        async def cancel_pipeline():
            """Cancel current pipeline execution."""
            with self._lock:
                if self.current_status["stage"] not in ["running"]:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Cannot cancel pipeline in {self.current_status['stage']} state",
                    )

                try:
                    # Set cancellation flag
                    self._cancel_requested.set()

                    # Update progress state
                    self.current_status.update(
                        {
                            "stage": "canceling",
                            "success": False,
                            "error": "Pipeline execution was canceled",
                        }
                    )

                    # Broadcast cancellation to connected clients
                    await self._broadcast_update()

                    return {
                        "status": "cancel_requested",
                        "message": "Pipeline cancellation requested",
                        "stage": "canceling",
                    }

                except Exception as e:
                    raise HTTPException(
                        status_code=500, detail=f"Cancel failed: {str(e)}"
                    )

        @self.app.delete("/pipeline/clear")
        async def clear_outputs():
            """Clear output data and reset to idle state."""
            with self._lock:
                try:
                    # Reset to idle state
                    self.current_status = {
                        "stage": "idle",
                        "total_transforms": 0,
                        "completed_transforms": 0,
                        "current_transform": None,
                        "success": True,
                        "error": None,
                        "progress_percentage": 0,
                    }

                    # Broadcast clear to connected clients
                    await self._broadcast_update()

                    return {
                        "status": "outputs_cleared",
                        "message": "Output data cleared and pipeline reset to idle",
                        "stage": "idle",
                    }

                except Exception as e:
                    raise HTTPException(
                        status_code=500, detail=f"Clear failed: {str(e)}"
                    )

    async def _broadcast_update(self):
        """Broadcast current status to all WebSocket connections."""
        if not self.connections:
            return

        message = json.dumps(self.current_status)
        disconnected = []

        for connection in self.connections:
            try:
                await connection.send_text(message)
            except Exception:
                # Connection was closed, mark for removal
                disconnected.append(connection)

        # Remove disconnected connections
        for connection in disconnected:
            self.connections.remove(connection)

    def _update_status(self, **kwargs):
        """Update current status and broadcast to connected clients."""
        self.current_status.update(kwargs)

        # Calculate progress percentage
        if self.current_status["total_transforms"] > 0:
            self.current_status["progress_percentage"] = (
                self.current_status["completed_transforms"]
                / self.current_status["total_transforms"]
            ) * 100

        # Schedule broadcast using thread-safe approach
        self._schedule_broadcast()

    def _schedule_broadcast(self):
        """Schedule a broadcast update in a thread-safe way."""
        if not self.connections:
            return

        # Add update to queue for processing by WebSocket connections
        try:
            self._broadcast_queue.put_nowait(self.current_status.copy())
        except queue.Full:
            # Queue is full, skip this update
            pass

    def pre_pipe_execute(self, pipe: TransformPipe):
        """Initialize progress tracking before pipeline execution begins.

        Args:
            pipe: The transform pipeline about to be executed.
        """
        total_transforms = len(pipe.get_transforms())
        self._update_status(
            stage="running",
            total_transforms=total_transforms,
            completed_transforms=0,
            current_transform=None,
            success=True,
            error=None,
        )

    def pre_node_execute(self, node: AbstractExecuteTransform):
        """Update progress before each transform execution.

        Args:
            node: The transform that is about to be executed.
        """
        self._update_status(
            current_transform={
                "name": node.get_name(),
                "docs": node.get_docs(),
                "status": "running",
            }
        )

    def post_node_execute(
        self, node: AbstractExecuteTransform, success: bool, error: Optional[Exception]
    ):
        """Update progress after each transform execution.

        Args:
            node: The transform that just finished executing.
            success: Whether the transform executed successfully.
            error: Exception that occurred during execution, if any.
        """
        completed = self.current_status["completed_transforms"] + 1

        if success:
            self._update_status(
                completed_transforms=completed,
                current_transform={
                    "name": node.get_name(),
                    "docs": node.get_docs(),
                    "status": "completed",
                },
            )
        else:
            self._update_status(
                success=False,
                error=str(error) if error else "Unknown error",
                current_transform={
                    "name": node.get_name(),
                    "docs": node.get_docs(),
                    "status": "failed",
                    "error": str(error) if error else "Unknown error",
                },
            )

    def post_pipe_execute(
        self, pipe: TransformPipe, success: bool, error: Optional[Exception]
    ):
        """Finalize progress tracking after pipeline execution completes.

        Args:
            pipe: The transform pipeline that finished executing.
            success: Whether the entire pipeline executed successfully.
            error: Exception that occurred during pipeline execution, if any.
        """
        if success:
            self._update_status(
                stage="completed",
                completed_transforms=self.current_status["total_transforms"],
                current_transform=None,
            )
        else:
            self._update_status(
                stage="failed",
                success=False,
                error=str(error) if error else "Pipeline execution failed",
                current_transform=None,
            )

    def is_cancellation_requested(self) -> bool:
        """Check if pipeline cancellation has been requested.

        Returns:
            True if cancellation was requested via the /pipeline/cancel endpoint.
        """
        return self._cancel_requested.is_set()

    def is_reset_requested(self) -> bool:
        """Check if pipeline reset has been requested.

        Returns:
            True if reset was requested via the /pipeline/reset endpoint.
        """
        return self._reset_requested.is_set()
