from pond.hooks.abstract_hook import AbstractHook
from pond.runners.abstract_runner import AbstractRunner
from pond.state import State
from pond.transforms.transform_pipe import TransformPipe


class SequentialRunner(AbstractRunner):
    def __init__(self):
        super().__init__()

    def run(self, state: State, pipe: TransformPipe, hooks: list[AbstractHook]):
        for hook in hooks:
            hook.initialize(state.root_type)
            hook.pre_pipe_execute(pipe)
        error = None
        success = True
        for transform in pipe.get_transforms():
            for hook in hooks:
                hook.pre_node_execute(transform)
            try:
                for unit in transform.get_execute_units(state):
                    unit.execute_on(state)
            except Exception as e:
                error = e
                success = False
            for hook in hooks:
                hook.post_node_execute(transform, success, error)
            if error is not None:
                raise RuntimeError(
                    f"Failed at transform {transform.get_name()}"
                ) from error
        for hook in hooks:
            hook.post_pipe_execute(pipe, success, error)
