import dataclasses
import json


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        if hasattr(o, "to_json"):
            return o.to_json()
        return super().default(o)


def set_span_attribute(span, name, value):
    if value is not None:
        if value != "":
            span.set_attribute(name, value)
    return


def with_tracer_wrapper(func):
    def _with_tracer(tracer, to_wrap):
        def wrapper(wrapped, instance, args, kwargs):
            # prevent double wrapping
            if hasattr(wrapped, "__wrapped__"):
                return wrapped(*args, **kwargs)

            return func(tracer, to_wrap, wrapped, instance, args, kwargs)

        return wrapper

    return _with_tracer
