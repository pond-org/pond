from pond.lens import LensPath, get_cleaned_path
from pond.transforms.abstract_transform import (
    AbstractExecuteTransform,
    AbstractTransform,
)

# from pond.transform import Transform


class TransformPipe(AbstractTransform):
    def __init__(
        self,
        transforms: list[AbstractTransform],
        input: list[str] | str = [],
        output: list[str] | str = [],
        root_path: str = "catalog",
    ):
        self.transforms = transforms
        self.inputs = [
            get_cleaned_path(i, root_path)
            for i in (input if isinstance(input, list) else [input])
        ]
        self.outputs = [
            get_cleaned_path(o, root_path)
            for o in (output if isinstance(output, list) else [output])
        ]
        # Check if we can run these transforms in order
        produced: list[LensPath] = list(self.inputs)
        for transform in transforms:
            for i in transform.get_inputs():
                # assert i in produced
                assert any(i.subset_of(p) for p in produced), (
                    f"Input {i.to_path()} not in inputs or produced!"
                )
            for o in transform.get_outputs():
                # assert o not in produced
                assert all(not o.subset_of(p) for p in produced), (
                    f"Output {o.to_path()} already in inputs or produced!"
                )
                produced.append(o)
        for o in self.outputs:
            assert o in produced, (
                f"{o.to_path()} not in {[r.to_path() for r in self.outputs]}"
            )

    def get_inputs(self) -> list[LensPath]:
        return self.inputs

    def get_outputs(self) -> list[LensPath]:
        return self.outputs

    def get_transforms(self) -> list[AbstractExecuteTransform]:
        return sum([t.get_transforms() for t in self.transforms], [])
