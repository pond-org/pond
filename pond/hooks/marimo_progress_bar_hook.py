from typing import Optional

import marimo as mo

from pond.hooks.abstract_hook import AbstractHook
from pond.transforms.transform_pipe import TransformPipe
from pond.transforms.abstract_transform import AbstractExecuteTransform


class MarimoProgressBarHook(AbstractHook):
    def __init__(self):
        super().__init__()
        self.progress_bar = None

    def pre_pipe_execute(self, pipe: TransformPipe):
        nbr_transforms = len(pipe.get_transforms())
        self.progress_bar = mo.status.progress_bar(
            total=nbr_transforms,
            title="Running pipeline",
            subtitle="Running first node",
        )

    def post_node_execute(
        self, node: AbstractExecuteTransform, success: bool, error: Optional[Exception]
    ):
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
        if success:
            self.progress_bar.progress.update(
                increment=0,
                title="Run successful",
                subtitle="Finished running pipeline",
            )
        else:
            self.progress_bar.progress.update(increment=0, title="Failed")
        self.progress_bar.progress.close()
