import json
from typing import Any, cast

import pytest

from openinference.instrumentation import (
    Annotation,
    AnnotationScope,
    get_annotation_attributes,
    get_evaluation_attributes,
)


def test_get_annotation_attributes_serializes_all_fields() -> None:
    attributes = get_annotation_attributes(
        annotations=[
            Annotation(
                name="hallucination",
                score=0,
                label="hallucinated",
                explanation="The claim is unsupported.",
                annotator_kind="LLM",
                identifier="judge-v2",
                metadata={"rubric_version": 2},
            )
        ]
    )

    assert attributes == {
        "annotations.0.annotation.name": "hallucination",
        "annotations.0.annotation.score": 0,
        "annotations.0.annotation.label": "hallucinated",
        "annotations.0.annotation.explanation": "The claim is unsupported.",
        "annotations.0.annotation.annotator_kind": "LLM",
        "annotations.0.annotation.identifier": "judge-v2",
        "annotations.0.annotation.metadata": json.dumps({"rubric_version": 2}),
    }


@pytest.mark.parametrize(
    ("scope", "prefix"),
    [
        ("span", "evaluations"),
        ("trace", "trace.evaluations"),
        ("session", "session.evaluations"),
    ],
)
def test_get_evaluation_attributes_supports_every_scope(
    scope: AnnotationScope,
    prefix: str,
) -> None:
    attributes = get_evaluation_attributes(
        evaluations=[
            Annotation(name="correctness", score=0.9),
            Annotation(name="style", label="concise"),
        ],
        scope=scope,
    )

    assert attributes == {
        f"{prefix}.0.evaluation.name": "correctness",
        f"{prefix}.0.evaluation.score": 0.9,
        f"{prefix}.1.evaluation.name": "style",
        f"{prefix}.1.evaluation.label": "concise",
    }


def test_annotation_attribute_forms_are_composable_and_preserve_metadata_strings() -> None:
    attributes = {
        **get_annotation_attributes(
            annotations=[Annotation(name="quality", explanation="Looks good")],
            scope="trace",
        ),
        **get_evaluation_attributes(
            evaluations=[Annotation(name="quality", score=1, metadata='{"source":"review"}')],
            scope="session",
        ),
    }

    assert attributes == {
        "trace.annotations.0.annotation.name": "quality",
        "trace.annotations.0.annotation.explanation": "Looks good",
        "session.evaluations.0.evaluation.name": "quality",
        "session.evaluations.0.evaluation.score": 1,
        "session.evaluations.0.evaluation.metadata": '{"source":"review"}',
    }


def test_get_annotation_attributes_supports_session_scope() -> None:
    assert get_annotation_attributes(
        annotations=[Annotation(name="coherence", label="coherent")],
        scope="session",
    ) == {
        "session.annotations.0.annotation.name": "coherence",
        "session.annotations.0.annotation.label": "coherent",
    }


def test_get_annotation_attributes_accepts_an_empty_collection() -> None:
    assert get_annotation_attributes(annotations=[]) == {}


@pytest.mark.parametrize(
    "annotations",
    [
        [{"score": 1}],
        [{"name": "correctness"}],
        ["not-an-annotation-object"],
    ],
)
def test_get_evaluation_attributes_rejects_invalid_annotations(annotations: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        get_evaluation_attributes(evaluations=cast(Any, annotations))


def test_get_evaluation_attributes_rejects_invalid_scope() -> None:
    with pytest.raises(ValueError, match="Invalid annotation terminology or scope"):
        get_evaluation_attributes(
            evaluations=[Annotation(name="correctness", score=1)],
            scope=cast(Any, "conversation"),
        )
