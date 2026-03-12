import json
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel

from openinference.instrumentation._attributes import _json_serialize


class AttributeKey(Enum):
    INTEGER_ENUM = 2
    STRING_ENUM = "str_enum"


class AttributesModel(BaseModel):
    attributes: dict[Any, Any]


def test_json_serialize_pydantic_model_with_non_json_dict_keys() -> None:
    model = AttributesModel(
        attributes={
            AttributeKey.INTEGER_ENUM: "an integer enum key",
            AttributeKey.STRING_ENUM: "a string enum key",
            UUID("00000000-0000-0000-0000-000000000011"): "a uuid key",
            "str": "a string key",
            9: "an integer key",
            0.1: "a float key",
            True: "a boolean key",
            None: "a null key",
        },
    )

    expected = {
        "attributes": {
            "2": "an integer enum key",
            "str_enum": "a string enum key",
            "00000000-0000-0000-0000-000000000011": "a uuid key",
            "str": "a string key",
            "9": "an integer key",
            "0.1": "a float key",
            "true": "a boolean key",
            "None": "a null key",
        },
    }

    assert json.loads(_json_serialize(model)) == expected
