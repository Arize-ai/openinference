import json
import sys

import anthropic
from pydantic import RootModel


class RawMessageStreamEvent(RootModel[anthropic.types.RawMessageStreamEvent]):
    root: anthropic.types.RawMessageStreamEvent


if __name__ == "__main__":
    file_path = "anthropic_schema.json" if len(sys.argv) < 2 else sys.argv[1]
    with open(file_path, "w") as f:
        json.dump(RawMessageStreamEvent.model_json_schema(), f, indent=2)
