import argparse

from pond import State

from pond.catalogs.iceberg_catalog import IcebergCatalog
from pond.catalogs.lance_catalog import LanceCatalog
from pond.hooks.ui_hook import UIHook
from pond.runners.sequential_runner import SequentialRunner
from pond.runners.parallel_runner import ParallelRunner
from pond.volume import load_volume_protocol_args

from example.catalog import Catalog
from example.pipeline import heightmap_pipe


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
    pipeline = heightmap_pipe()
    runner.run(state, pipeline, hooks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalog", help="Catalog type: iceberg or hamilton", default="iceberg"
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
