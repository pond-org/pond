from pond.abstract_catalog import (
    # TypeField,
    LensPath,
)


def test_lens_path():
    first = LensPath.from_path("values.drives")
    second = LensPath.from_path("values.drives[0]")

    assert first != second
    assert second.subset_of(first)
    assert not first.subset_of(second)

    assert second.subset_of(second)
    assert first.subset_of(first)
    assert first == first
    assert second == second

    third = LensPath.from_path("values")
    assert first.subset_of(third)
    assert second.subset_of(third)
    assert first != third
