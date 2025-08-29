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
import os
import argparse

os.environ["PYICEBERG_HOME"] = os.getcwd()  # noqa: E402

from example.catalog import Catalog
from example.pipeline import heightmap_pipe
from pond import State, pipe, index_files
from pond.catalogs.iceberg_catalog import IcebergCatalog
from pond.catalogs.lance_catalog import LanceCatalog
from pond.hooks.ui_hook import UIHook
from pond.runners.parallel_runner import ParallelRunner
from pond.runners.sequential_runner import SequentialRunner
from pond.volume import load_volume_protocol_args


def main(args):
    match args.catalog:
        case "iceberg":
            catalog = IcebergCatalog(name="default")
        case "lance":
            catalog = LanceCatalog(db_path="lance_catalog")
        case _:
            raise ValueError(f"{args.catalog} not a valid catalog")
    volume_args = load_volume_protocol_args()
    state = State(Catalog, catalog, volume_protocol_args=volume_args)
    hooks = []
    if args.ui:
        hooks.append(UIHook(1, "nils", "pond"))
    match args.runner:
        case "sequential":
            runner = SequentialRunner()
        case "parallel":
            runner = ParallelRunner()
        case _:
            raise ValueError(f"{args.runner} not a valid runner")
    state["params.res"] = 4.0
    pipeline = pipe(
        [
            index_files(Catalog, "cloud_files"),
            heightmap_pipe(),
        ],
        input="params",
        output=["heightmap_plot", "bounds"],
    )
    runner.run(state, pipeline, hooks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalog", help="Catalog type: iceberg or lance", default="iceberg"
    )
    parser.add_argument(
        "--runner", help="Runner type: sequential or parallel", default="parallel"
    )
    parser.add_argument(
        "--ui",
        help="Publish UI, requires running hamilton UI",
        action=argparse.BooleanOptionalAction,
    )
    args = parser.parse_args()
    main(args)
