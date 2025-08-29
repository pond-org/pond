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
from pond.catalogs.abstract_catalog import LensPath


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
