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
#       item_namespaces: [message]   # flattened_* only: any `message.*` attribute
#       item_attributes: [tool.name] # flattened_* only: exactly these attributes
#       also_accepts: [int]          # other OTLP types consumers must accept for this key
#       constant_key: llm.prompt_template # key a published constant is named after, when not its own

encodings := {"flattened_list", "flattened_object", "json"}

flattened_encodings := {"flattened_list", "flattened_object"}

oi_annotations(attr) := object.get(object.get(attr, "annotations", {}), "openinference", {})

encoding(attr) := object.get(oi_annotations(attr), "encoding", "")

list_annotations := {"item_namespaces", "item_attributes", "also_accepts"}

# A list annotation written as a scalar (`item_namespaces: message`) reads as an
# empty list, so that the rules below can iterate it; it is reported on its own.
list_annotation(attr, name) := value if {
    value := object.get(oi_annotations(attr), name, [])
    is_array(value)
} else := []

item_namespaces(attr) := list_annotation(attr, "item_namespaces")

item_attributes(attr) := list_annotation(attr, "item_attributes")

attribute_keys contains attr.key if some attr in input.registry.attributes

# A key with no dot (`metadata`, `annotations`) has no namespace, so it does not
# make its own name one: `annotations` is not the `annotation.*` namespace.
root_namespaces contains ns if {
    some attr in input.registry.attributes
    parts := split(attr.key, ".")
    count(parts) > 1
    ns := parts[0]
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

# A flattened attribute says which attributes are found under it: whole
# namespaces (`item_namespaces`), individual attributes (`item_attributes`) when
# only part of a namespace applies, or both.
deny contains finding if {
    some attr in input.registry.attributes
    encoding(attr) in flattened_encodings
    count(item_namespaces(attr)) + count(item_attributes(attr)) == 0

    finding := {
        "id": "openinference_items_missing",
        "context": {"attribute": attr.key},
        "message": sprintf("Attribute '%s' is flattened and must list annotations.openinference.item_namespaces or item_attributes.", [attr.key]),
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

# ...as must each of those attributes.
deny contains finding if {
    some attr in input.registry.attributes
    some item in item_attributes(attr)
    not item in attribute_keys

    finding := {
        "id": "openinference_item_attribute_unknown",
        "context": {"attribute": attr.key, "item_attribute": item},
        "message": sprintf("Attribute '%s' lists item attribute '%s', which the registry does not define.", [attr.key, item]),
        "level": "violation",
    }
}

# A list cannot be an item of itself.
deny contains finding if {
    some attr in input.registry.attributes
    attr.key in item_attributes(attr)

    finding := {
        "id": "openinference_item_attribute_self",
        "context": {"attribute": attr.key},
        "message": sprintf("Attribute '%s' lists itself in annotations.openinference.item_attributes.", [attr.key]),
        "level": "violation",
    }
}

# `also_accepts` records what the attribute type cannot: that some
# instrumentations record another primitive (e.g. an integer `document.id`).
also_accepts_types := {"string", "int", "double", "boolean"}

deny contains finding if {
    some attr in input.registry.attributes
    some accepted in list_annotation(attr, "also_accepts")
    not accepted in also_accepts_types

    finding := {
        "id": "openinference_also_accepts_unknown",
        "context": {"attribute": attr.key, "also_accepts": accepted},
        "message": sprintf("Attribute '%s' lists unknown type '%s' in annotations.openinference.also_accepts. Expected one of: string, int, double, boolean.", [attr.key, accepted]),
        "level": "violation",
    }
}

# The list annotations are lists, even of one item.
deny contains finding if {
    some attr in input.registry.attributes
    some name in list_annotations
    value := object.get(oi_annotations(attr), name, [])
    not is_array(value)

    finding := {
        "id": "openinference_annotation_not_a_list",
        "context": {"attribute": attr.key, "annotation": name},
        "message": sprintf("Attribute '%s' must write annotations.openinference.%s as a list.", [attr.key, name]),
        "level": "violation",
    }
}
