import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os

    os.environ["PYICEBERG_HOME"] = os.getcwd()  # noqa: E402

    from example.catalog import Catalog
    from example.pipeline import heightmap_pipe
    from pond import State
    from pond.catalogs.iceberg_catalog import IcebergCatalog
    from pond.hooks.marimo_progress_bar_hook import MarimoProgressBarHook
    from pond.runners.parallel_runner import ParallelRunner
    from pond.volume import load_volume_protocol_args

    return (
        Catalog,
        IcebergCatalog,
        MarimoProgressBarHook,
        ParallelRunner,
        State,
        heightmap_pipe,
        load_volume_protocol_args,
        mo,
    )


@app.cell
def _(mo):
    res = mo.ui.slider(start=1, stop=20, label="Resolution", value=3)
    res
    return (res,)


@app.cell
def _(
    Catalog,
    IcebergCatalog,
    MarimoProgressBarHook,
    ParallelRunner,
    State,
    heightmap_pipe,
    load_volume_protocol_args,
    res,
):
    volume_args = load_volume_protocol_args()
    catalog = IcebergCatalog(name="default")
    runner = ParallelRunner()
    state = State(Catalog, catalog, volume_protocol_args=volume_args)
    hooks = [MarimoProgressBarHook()]
    state["params.res"] = float(res.value)
    pipeline = heightmap_pipe()
    runner.run(state, pipeline, hooks)
    return (state,)


@app.cell
def _(state):
    iceberg_catalog = state.catalog.catalog
    return (iceberg_catalog,)


@app.cell
def _(iceberg_catalog):
    iceberg_catalog.load_table("catalog.cloud_files").scan().to_arrow()
    return


if __name__ == "__main__":
    app.run()
