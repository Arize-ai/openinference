package after_resolution

import rego.v1

# OpenInference-specific rules, run alongside the shared OpenTelemetry policies.
#
# OTLP span attributes cannot hold objects, so OpenInference flattens structured
# values into one attribute per leaf (`llm.input_messages.0.message.role`) or
# serializes them into a JSON string. The registry records which encoding an
# attribute uses in `annotations.openinference`, so that consumers and code
# generators can rebuild the structure:
#
#   annotations:
#     openinference:
#       encoding: flattened_list | flattened_object | json
#       item_namespaces: [message]   # flattened_* only

encodings := {"flattened_list", "flattened_object", "json"}

flattened_encodings := {"flattened_list", "flattened_object"}

oi_annotations(attr) := object.get(object.get(attr, "annotations", {}), "openinference", {})

encoding(attr) := object.get(oi_annotations(attr), "encoding", "")

item_namespaces(attr) := object.get(oi_annotations(attr), "item_namespaces", [])

root_namespaces contains ns if {
    some attr in input.registry.attributes
    ns := split(attr.key, ".")[0]
}

# A `template[any]` attribute is a key prefix with arbitrary suffixes. In this
# registry that always means a flattened list or object.
deny contains finding if {
    some attr in input.registry.attributes
    attr.type == "template[any]"
    not encoding(attr) in flattened_encodings

    finding := {
        "id": "openinference_flattened_encoding_missing",
        "context": {"attribute": attr.key},
        "message": sprintf("Attribute '%s' has type 'template[any]' and must set annotations.openinference.encoding to 'flattened_list' or 'flattened_object'.", [attr.key]),
        "level": "violation",
    }
}

deny contains finding if {
    some attr in input.registry.attributes
    enc := encoding(attr)
    enc != ""
    not enc in encodings

    finding := {
        "id": "openinference_encoding_unknown",
        "context": {"attribute": attr.key, "encoding": enc},
        "message": sprintf("Attribute '%s' has unknown annotations.openinference.encoding '%s'. Expected one of: flattened_list, flattened_object, json.", [attr.key, enc]),
        "level": "violation",
    }
}

# The flattened encodings only make sense on a key prefix.
deny contains finding if {
    some attr in input.registry.attributes
    encoding(attr) in flattened_encodings
    attr.type != "template[any]"

    finding := {
        "id": "openinference_flattened_encoding_type",
        "context": {"attribute": attr.key, "type": attr.type},
        "message": sprintf("Attribute '%s' is annotated as flattened and must have type 'template[any]'.", [attr.key]),
        "level": "violation",
    }
}

# A JSON string is a string.
deny contains finding if {
    some attr in input.registry.attributes
    encoding(attr) == "json"
    attr.type != "string"

    finding := {
        "id": "openinference_json_encoding_type",
        "context": {"attribute": attr.key, "type": attr.type},
        "message": sprintf("Attribute '%s' is annotated as a JSON string and must have type 'string'.", [attr.key]),
        "level": "violation",
    }
}

# A flattened attribute names the namespace(s) of the attributes found under it.
deny contains finding if {
    some attr in input.registry.attributes
    encoding(attr) in flattened_encodings
    count(item_namespaces(attr)) == 0

    finding := {
        "id": "openinference_item_namespaces_missing",
        "context": {"attribute": attr.key},
        "message": sprintf("Attribute '%s' is flattened and must list annotations.openinference.item_namespaces.", [attr.key]),
        "level": "violation",
    }
}

# ...and each of those namespaces must exist in the registry.
deny contains finding if {
    some attr in input.registry.attributes
    some ns in item_namespaces(attr)
    not ns in root_namespaces

    finding := {
        "id": "openinference_item_namespace_unknown",
        "context": {"attribute": attr.key, "item_namespace": ns},
        "message": sprintf("Attribute '%s' lists item namespace '%s', but the registry defines no '%s.*' attributes.", [attr.key, ns, ns]),
        "level": "violation",
    }
}
