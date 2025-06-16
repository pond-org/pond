from typing import Any
from concurrent.futures import as_completed
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import multiprocessing
from multiprocessing import Event
from multiprocessing.context import ForkContext, SpawnContext

from loguru import logger

from pond.runners.abstract_runner import AbstractRunner
from pond.state import State
from pond.transforms.transform_pipe import TransformPipe
from pond.transforms.abstract_transform import AbstractExecuteUnit
from pond.hooks.abstract_hook import AbstractHook


def execute_unit(state: State, unit: AbstractExecuteUnit, t: int, lock):
    args = unit.load_inputs(state)
    rtns = unit.run(args)
    values = unit.save_outputs(state, rtns)
    with lock:
        unit.commit(state, values)
    return unit, t


class ParallelRunner(AbstractRunner):
    def __init__(self):
        super().__init__()

    def run(self, state: State, pipe: TransformPipe, hooks: list[AbstractHook]):
        for hook in hooks:
            hook.initialize(state.root_type)
            hook.pre_pipe_execute(pipe)
        error = None
        success = True
        transforms = dict(enumerate(pipe.get_transforms()))
        dependencies = {
            t: transform.get_inputs() for t, transform in transforms.items()
        }
        todo = set(transforms.keys())
        inputs = pipe.get_inputs()
        produced = list(inputs)
        futures = set()
        # done_noes = set()

        execute_units_finished = {}

        with multiprocessing.Manager() as m:
            lock = m.Lock()
            with ProcessPoolExecutor(
                max_workers=10,
                # mp_context=ForkContext(),
                mp_context=SpawnContext(),
            ) as pool:
                while True:
                    ready = {
                        t
                        for t in todo
                        if all(
                            any(i.subset_of(p) for p in produced)
                            for i in dependencies[t]
                        )
                    }
                    logger.info(f"Found {len(ready)} nodes to execute")
                    todo -= ready
                    for t in ready:
                        transform = transforms[t]
                        for hook in hooks:
                            hook.pre_node_execute(transform)
                        execute_units = transform.get_execute_units(state)
                        execute_units_finished[t] = len(execute_units)
                        for unit in execute_units:
                            futures.add(
                                pool.submit(
                                    execute_unit,
                                    state,
                                    unit,
                                    t,
                                    lock,
                                )
                            )

                    if not futures:
                        logger.info("No futures left in pool, stopping...")
                        break

                    done, futures = wait(
                        futures, return_when=FIRST_COMPLETED, timeout=0.1
                    )

                    for future in done:
                        try:
                            # unit, rtns, t = future.result()
                            unit, t = future.result()
                            # unit.save_outputs(state, rtns)
                        except Exception as e:
                            error = e
                            success = False
                            for hook in hooks:
                                hook.post_node_execute(transform, success, error)
                                hook.post_pipe_execute(pipe, success, error)
                            raise RuntimeError(
                                f"Failed at transform {transform.get_name()}"
                            ) from error

                        execute_units_finished[t] = execute_units_finished[t] - 1
                        if execute_units_finished[t] == 0:
                            transform = transforms[t]
                            for hook in hooks:
                                hook.post_node_execute(transform, success, error)
                            for o in transform.get_outputs():
                                produced.append(o)

        for hook in hooks:
            hook.post_pipe_execute(pipe, success, error)
