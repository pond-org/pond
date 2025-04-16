import os
import yaml
from typing import Any


def load_volume_protocol_args() -> dict[str, Any]:
    volume_config_file = os.path.join(os.getcwd(), ".fsspec.yaml")
    with open(volume_config_file) as f:
        volume_protocol_args = yaml.safe_load(f)
    return volume_protocol_args
