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
from pond.lens import LensPath, get_cleaned_path
from pond.transforms.abstract_transform import (
    AbstractExecuteTransform,
    AbstractTransform,
)

# from pond.transform import Transform


class TransformPipe(AbstractTransform):
    """Pipeline container for composing multiple transforms into a single unit.

    TransformPipe allows you to group transforms together with explicit input and
    output specifications. It validates that the transforms can execute in sequence
    by ensuring all required inputs are available before each transform runs.

    Key Features:
    - Validates transform execution order at construction time
    - Ensures data dependencies are satisfied
    - Provides unified input/output interface for the entire pipeline
    - Supports both nested pipelines and individual transforms
    - Flattens nested transform hierarchies for execution

    Attributes:
        transforms: List of transforms that make up this pipeline.
        inputs: List of LensPath objects representing pipeline inputs.
        outputs: List of LensPath objects representing pipeline outputs.

    Note:
        The constructor performs validation to ensure the pipeline is executable:
        - All transform inputs must be available from pipeline inputs or previous outputs
        - No transform can produce an output that already exists
        - All specified pipeline outputs must be produced by some transform
    """

    def __init__(
        self,
        transforms: list[AbstractTransform],
        input: list[str] | str = [],
        output: list[str] | str = [],
        root_path: str = "catalog",
    ):
        """Initialize a TransformPipe with validation.

        Args:
            transforms: List of transforms to compose into a pipeline.
                Can include other TransformPipe instances for nesting.
            input: Input path(s) for the pipeline. Can be a single string
                or list of strings. These represent data that must be
                provided externally to run the pipeline.
            output: Output path(s) for the pipeline. Can be a single string
                or list of strings. These represent the final results
                produced by the pipeline.
            root_path: Root path name for path resolution. Defaults to "catalog".

        Raises:
            AssertionError: If pipeline validation fails:
                - Transform input not available from inputs or previous outputs
                - Transform output conflicts with existing data
                - Pipeline output not produced by any transform

        Note:
            Validation ensures the pipeline can execute successfully:
            1. Converts string paths to LensPath objects
            2. Checks each transform's inputs are available
            3. Verifies no output conflicts exist
            4. Confirms all pipeline outputs are produced
        """
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
        """Get the input paths required by this pipeline.

        Returns:
            List of LensPath objects representing data that must be
            provided externally to execute this pipeline.

        Note:
            These are the inputs specified during pipeline construction,
            not the inputs of individual transforms within the pipeline.
        """
        return self.inputs

    def get_outputs(self) -> list[LensPath]:
        """Get the output paths produced by this pipeline.

        Returns:
            List of LensPath objects representing the final data
            products generated by this pipeline.

        Note:
            These are the outputs specified during pipeline construction,
            representing the pipeline's public interface.
        """
        return self.outputs

    def get_transforms(self) -> list[AbstractExecuteTransform]:
        """Get all executable transforms in this pipeline, flattened.

        Returns:
            List of AbstractExecuteTransform objects representing all
            individual transforms that need to be executed, with any
            nested TransformPipe instances recursively flattened.

        Note:
            This method flattens the transform hierarchy:
            - Individual transforms are returned as-is
            - Nested TransformPipe instances are recursively expanded
            - The result is a flat list ready for execution by runners
        """
        return sum([t.get_transforms() for t in self.transforms], [])
