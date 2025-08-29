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
from typing import Optional

import marimo as mo

from pond.hooks.abstract_hook import AbstractHook
from pond.transforms.transform_pipe import TransformPipe
from pond.transforms.abstract_transform import AbstractExecuteTransform


class MarimoProgressBarHook(AbstractHook):
    """Progress bar hook for Marimo notebook integration.

    This hook provides visual progress tracking for PyPond pipeline execution
    within Marimo notebooks. It displays a progress bar that updates as
    transforms complete, showing current status and any errors that occur.

    Attributes:
        progress_bar: Marimo progress bar widget for visual feedback.

    Note:
        This hook is specifically designed for use within Marimo notebooks
        and requires the marimo package. The progress bar automatically
        tracks pipeline execution and provides real-time feedback to users.
    """

    def __init__(self):
        """Initialize the Marimo progress bar hook.

        Sets up the hook with no active progress bar initially.
        The progress bar will be created when pipeline execution begins.
        """
        super().__init__()
        self.progress_bar = None

    def pre_pipe_execute(self, pipe: TransformPipe):
        """Initialize progress bar before pipeline execution begins.

        Args:
            pipe: The transform pipeline about to be executed.

        Note:
            Creates a new Marimo progress bar widget with the total number
            of transforms to be executed. The progress bar shows an initial
            status indicating the pipeline is starting.
        """
        nbr_transforms = len(pipe.get_transforms())
        self.progress_bar = mo.status.progress_bar(
            total=nbr_transforms,
            title="Running pipeline",
            subtitle="Running first node",
        )

    def post_node_execute(
        self, node: AbstractExecuteTransform, success: bool, error: Optional[Exception]
    ):
        """Update progress bar after each transform execution.

        Args:
            node: The transform that just finished executing.
            success: Whether the transform executed successfully.
            error: Exception that occurred during execution, if any.

        Note:
            Updates the progress bar with current status:
            - Success: Shows "Finished {transform_name}" with running status
            - Failure: Shows "Error on {transform_name}" with failed status

            The progress automatically advances by one step with each update.
        """
        if success:
            self.progress_bar.progress.update(
                title="Running", subtitle=f"Finished {node.get_name()}"
            )
        else:
            self.progress_bar.progress.update(
                title="Failed", subtitle=f"Error on {node.get_name()}"
            )

    def post_pipe_execute(
        self, pipe: TransformPipe, success: bool, error: Optional[Exception]
    ):
        """Finalize progress bar after pipeline execution completes.

        Args:
            pipe: The transform pipeline that finished executing.
            success: Whether the entire pipeline executed successfully.
            error: Exception that occurred during pipeline execution, if any.

        Note:
            Updates the progress bar with final status and closes it:
            - Success: Shows "Run successful" with completion message
            - Failure: Shows "Failed" status

            Always closes the progress bar to clean up the UI widget
            regardless of execution outcome.
        """
        if success:
            self.progress_bar.progress.update(
                increment=0,
                title="Run successful",
                subtitle="Finished running pipeline",
            )
        else:
            self.progress_bar.progress.update(increment=0, title="Failed")
        self.progress_bar.progress.close()
