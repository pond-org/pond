from pond import State

from pond.catalogs.iceberg_catalog import IcebergCatalog
from pond.catalogs.lance_catalog import LanceCatalog
from pond.hooks.ui_hook import UIHook
from pond.runners.sequential_runner import SequentialRunner
from pond.runners.parallel_runner import ParallelRunner
from pond.volume import load_volume_protocol_args

from example.catalog import Catalog
from example.pipeline import heightmap_pipe


def main():
    catalog = IcebergCatalog(name="default")
    # catalog = LanceCatalog(db_path="lance_catalog")
    volume_args = load_volume_protocol_args()
    state = State(Catalog, catalog, volume_protocol_args=volume_args)
    ui_client = UIHook(1, "nils", "pond")
    # runner = SequentialRunner()
    runner = ParallelRunner()
    state["params.res"] = 4.0
    pipeline = heightmap_pipe()
    runner.run(state, pipeline, [ui_client])


if __name__ == "__main__":
    main()
