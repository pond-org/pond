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
from typing import Any

import yaml


def load_volume_protocol_args() -> dict[str, Any]:
    """Load fsspec protocol configuration from .fsspec.yaml file.

    Reads filesystem specification configuration from a .fsspec.yaml file in the
    current working directory. This configuration defines how different storage
    protocols should be handled by fsspec.

    Returns:
        Dict containing fsspec protocol configuration parameters. Keys are protocol
        names (e.g. 's3', 'gcs', 'file') and values are dictionaries of protocol-specific
        configuration options.

    Raises:
        FileNotFoundError: If .fsspec.yaml file does not exist in current directory.
        yaml.YAMLError: If the YAML file is malformed or cannot be parsed.

    Note:
        The function assumes the .fsspec.yaml file exists and contains valid YAML.
        The current working directory is used as the base path for locating the config file.
    """
    volume_config_file = os.path.join(os.getcwd(), ".fsspec.yaml")
    with open(volume_config_file) as f:
        volume_protocol_args = yaml.safe_load(f)
    return volume_protocol_args
