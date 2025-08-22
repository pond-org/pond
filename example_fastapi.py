"""Example usage of FastAPI integration with PyPond pipelines."""

import os
import threading

os.environ["PYICEBERG_HOME"] = os.getcwd()

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from example.catalog import Catalog
from example.pipeline import heightmap_pipe
from pond import State, fastapi_input, fastapi_output, pipe
from pond.api.progress_hook import FastAPIHook
from pond.catalogs.iceberg_catalog import IcebergCatalog
from pond.runners.parallel_runner import ParallelRunner
from pond.volume import load_volume_protocol_args


def main():
    """Example of full HTTP pipeline (input and output)."""
    print("=== Full HTTP Pipeline Example ===")

    # Create FastAPI app
    app = FastAPI(title="PyPond Pipeline API", version="1.0.0")

    # Add CORS middleware to allow web page requests
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, specify actual origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Set up pipeline components
    catalog = IcebergCatalog(name="default")
    volume_args = load_volume_protocol_args()
    state = State(Catalog, catalog, volume_protocol_args=volume_args)
    # runner = SequentialRunner()
    runner = ParallelRunner()

    # Create unified FastAPI hook that handles both progress and control
    fastapi_hook = FastAPIHook(app)

    # Create pipeline with shared FastAPI app
    pipeline = pipe(
        [
            fastapi_input(Catalog, input=["params", "cloud_files"], app=app),
            heightmap_pipe(),  # Same existing pipeline!
            fastapi_output(Catalog, output=["heightmap_plot", "bounds"], app=app),
        ]
    )

    # Start FastAPI server in background thread
    def run_server():
        uvicorn.run(app, host="localhost", port=8000, log_level="info")

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    print("Full HTTP pipeline starting on http://localhost:8000")
    print("Input endpoints:")
    print("  POST /input/params - JSON: {'res': 4.0}")
    print("  POST /input/cloud-files - File upload")
    print("Output endpoints:")
    print("  GET /output/heightmap-plot - Download PNG")
    print("  GET /output/bounds - Get bounds JSON")
    print("  GET /status - Check pipeline input status")
    print("  GET /output-status - Check pipeline output status")
    print("Progress endpoints:")
    print("  WebSocket /progress/ws - Real-time progress updates")
    print("  GET /progress/sse - Server-Sent Events progress stream")
    print("  GET /progress/status - Current progress status")
    print("Control endpoints:")
    print("  POST /pipeline/reset - Reset pipeline state")
    print("  POST /pipeline/cancel - Cancel current execution")
    print("  DELETE /pipeline/clear - Clear output data")
    print()
    print(
        "Server running in background, pipeline will start when inputs are provided..."
    )
    print(
        "You can now use the control endpoints to reset/cancel the pipeline as needed."
    )

    # Run the pipeline with unified Fastapi hook
    runner.run(state, pipeline, [fastapi_hook])

    print("\n" + "=" * 50)
    print("Pipeline completed successfully!")
    print("Results are now available at:")
    print("  GET /output/heightmap_plot - Download heightmap PNG")
    print("  GET /output/bounds - Get bounds JSON")
    print("  GET /output-status - Check what outputs are available")
    print("=" * 50)
    print("Server will continue running for result downloads...")
    print("Press Ctrl+C to stop the server")

    # Keep the server running after pipeline completion
    try:
        while True:
            import time

            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down server...")


if __name__ == "__main__":
    main()
