package after_resolution

import rego.v1

# Every OpenInference span is identified by `openinference.span.kind`.
deny contains finding if {
    some span in input.registry.spans
    not requires_span_kind(span)

    finding := {
        "id": "openinference_span_kind_not_required",
        "context": {},
        "message": sprintf("Span '%s' must reference 'openinference.span.kind' with requirement level 'required'.", [span.type]),
        "level": "violation",
        "signal_type": "span",
        "signal_name": span.type,
    }
}

requires_span_kind(span) if {
    some attr in span.attributes
    attr.key == "openinference.span.kind"
    attr.requirement_level == "required"
}
