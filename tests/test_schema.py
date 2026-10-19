import json

import pytest
from pydantic import ValidationError

from engine.prompt_builder import build_schema
from engine.schema import build_json_schema, build_record_model
from tests.conftest import envelope


def _walk(node, key):
    if isinstance(node, dict):
        if key in node:
            yield node[key]
        for v in node.values():
            yield from _walk(v, key)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v, key)


def test_schema_shape(prostate):
    schema = build_json_schema(prostate)
    assert schema["type"] == "object" and schema["additionalProperties"] is False
    assert list(schema["properties"]) == prostate.field_ids
    assert schema["required"] == prostate.field_ids
    # portable: no $ref / $defs / title anywhere
    assert not list(_walk(schema, "$ref")) and "$defs" not in schema and not list(_walk(schema, "title"))
    json.dumps(schema)  # serializable

    gp = schema["properties"]["gleason_primary"]
    assert list(gp["properties"]) == ["evidence", "status", "value", "hedging", "conflict"]
    assert gp["required"] == ["evidence", "status", "value", "hedging", "conflict"]
    assert gp["additionalProperties"] is False
    assert gp["properties"]["evidence"] == {"type": "array", "items": {"type": "string"}}
    assert gp["properties"]["status"]["enum"] == ["value", "explicitly_absent", "not_mentioned"]
    assert gp["properties"]["hedging"]["enum"] == ["definite", "probable", "possible", "indeterminate"]
    assert gp["properties"]["conflict"] == {"type": "boolean"}
    value = gp["properties"]["value"]
    assert {"enum": [1, 2, 3, 4, 5], "type": "integer"} in value["anyOf"] and {"type": "null"} in value["anyOf"]

    assert schema["properties"]["nodes_examined"]["properties"]["value"]["anyOf"][0] == {"type": "integer"}
    assert schema["properties"]["margin_location"]["properties"]["value"]["anyOf"][0] == {"type": "string"}
    assert schema["properties"]["tumor_percent_of_gland"]["properties"]["value"]["anyOf"][0] == {"type": "number"}
    assert schema["properties"]["pT"]["properties"]["value"]["anyOf"][0]["enum"] == prostate.field("pT").allowed

    td = schema["properties"]["tumor_dimension"]["properties"]
    assert list(td) == ["evidence", "status", "value", "unit", "hedging", "conflict"]
    assert td["unit"]["anyOf"][0] == {"enum": ["mm", "cm"], "type": "string"}


def test_prompt_builder_exposes_same_schema(prostate):
    assert build_schema(prostate) == build_json_schema(prostate)


def test_record_model_validates_and_rejects(prostate):
    Record = build_record_model(prostate)
    assert build_record_model(prostate) is Record            # cached per pack
    Env = Record.model_fields["gleason_primary"].annotation
    ok = Env.model_validate(envelope("value", 3, ["Gleason 3+4"]))
    assert ok.value == 3 and ok.status == "value"
    with pytest.raises(ValidationError):
        Env.model_validate(envelope("value", 7, ["x"]))              # not in allowed
    with pytest.raises(ValidationError):
        Env.model_validate(envelope("maybe", 3, ["x"]))              # bad status
    with pytest.raises(ValidationError):
        Env.model_validate({**envelope("value", 3, ["x"]), "extra": 1})   # extra key
    with pytest.raises(ValidationError):
        Env.model_validate({"status": "value", "value": 3})          # missing required keys
    Meas = Record.model_fields["tumor_dimension"].annotation
    m = Meas.model_validate(envelope("value", 1.5, ["1.5 cm"], unit="cm", measurement=True))
    assert m.unit == "cm"
    with pytest.raises(ValidationError):
        Meas.model_validate(envelope("value", 1.5, ["x"], unit="inch", measurement=True))
