from google.genai import types

from openinference.instrumentation.google_genai._utils import (
    _get_token_count_attributes_from_usage_metadata,
)
from openinference.semconv.trace import SpanAttributes


class TestGetTokenCountAttributesFromUsageMetadata:
    """Tests for _get_token_count_attributes_from_usage_metadata function."""

    def test_empty_usage_metadata_yields_no_attributes(self) -> None:
        """Empty usage metadata should not yield any attributes."""
        result = dict(
            _get_token_count_attributes_from_usage_metadata(
                types.GenerateContentResponseUsageMetadata()
            )
        )
        assert result == {}

    def test_total_token_count(self) -> None:
        """Should extract total token count."""
        result = dict(
            _get_token_count_attributes_from_usage_metadata(
                types.GenerateContentResponseUsageMetadata(total_token_count=100)
            )
        )
        assert result == {SpanAttributes.LLM_TOKEN_COUNT_TOTAL: 100}

    def test_prompt_token_count(self) -> None:
        """Should extract prompt token count."""
        result = dict(
            _get_token_count_attributes_from_usage_metadata(
                types.GenerateContentResponseUsageMetadata(prompt_token_count=50)
            )
        )
        assert result == {SpanAttributes.LLM_TOKEN_COUNT_PROMPT: 50}

    def test_prompt_token_count_with_tool_use(self) -> None:
        """Should sum prompt_token_count and tool_use_prompt_token_count."""
        result = dict(
            _get_token_count_attributes_from_usage_metadata(
                types.GenerateContentResponseUsageMetadata(
                    prompt_token_count=50, tool_use_prompt_token_count=30
                )
            )
        )
        assert result[SpanAttributes.LLM_TOKEN_COUNT_PROMPT] == 80

    def test_tool_use_prompt_token_count_only(self) -> None:
        """Should handle tool_use_prompt_token_count without base prompt_token_count."""
        result = dict(
            _get_token_count_attributes_from_usage_metadata(
                types.GenerateContentResponseUsageMetadata(tool_use_prompt_token_count=30)
            )
        )
        assert result == {SpanAttributes.LLM_TOKEN_COUNT_PROMPT: 30}

    def test_candidates_token_count(self) -> None:
        """Should extract candidates token count as completion."""
        result = dict(
            _get_token_count_attributes_from_usage_metadata(
                types.GenerateContentResponseUsageMetadata(candidates_token_count=100)
            )
        )
        assert result == {SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: 100}

    def test_thoughts_token_count(self) -> None:
        """Should extract thoughts token count and add to completion."""
        result = dict(
            _get_token_count_attributes_from_usage_metadata(
                types.GenerateContentResponseUsageMetadata(thoughts_token_count=50)
            )
        )
        assert result == {
            SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING: 50,
            SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: 50,
        }

    def test_completion_with_thoughts(self) -> None:
        """Should sum candidates_token_count and thoughts_token_count for completion."""
        result = dict(
            _get_token_count_attributes_from_usage_metadata(
                types.GenerateContentResponseUsageMetadata(
                    candidates_token_count=100, thoughts_token_count=50
                )
            )
        )
        assert result[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION] == 150
        assert result[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING] == 50

    def test_prompt_tokens_details_audio(self) -> None:
        """Should extract audio token count from prompt_tokens_details."""
        result = dict(
            _get_token_count_attributes_from_usage_metadata(
                types.GenerateContentResponseUsageMetadata(
                    prompt_tokens_details=[
                        types.ModalityTokenCount(
                            modality=types.MediaModality.TEXT, token_count=100
                        ),
                        types.ModalityTokenCount(
                            modality=types.MediaModality.AUDIO, token_count=25
                        ),
                    ]
                )
            )
        )
        assert result == {SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_AUDIO: 25}

    def test_candidates_tokens_details_audio(self) -> None:
        """Should extract audio token count from candidates_tokens_details."""
        result = dict(
            _get_token_count_attributes_from_usage_metadata(
                types.GenerateContentResponseUsageMetadata(
                    candidates_tokens_details=[
                        types.ModalityTokenCount(
                            modality=types.MediaModality.TEXT, token_count=100
                        ),
                        types.ModalityTokenCount(
                            modality=types.MediaModality.AUDIO, token_count=10
                        ),
                    ]
                )
            )
        )
        assert result == {SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_AUDIO: 10}

    def test_multiple_audio_modalities_are_summed(self) -> None:
        """Should sum multiple audio modality entries."""
        result = dict(
            _get_token_count_attributes_from_usage_metadata(
                types.GenerateContentResponseUsageMetadata(
                    prompt_tokens_details=[
                        types.ModalityTokenCount(
                            modality=types.MediaModality.AUDIO, token_count=10
                        ),
                        types.ModalityTokenCount(
                            modality=types.MediaModality.AUDIO, token_count=15
                        ),
                    ]
                )
            )
        )
        assert result == {SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_AUDIO: 25}

    def test_no_audio_modality_yields_no_audio_attributes(self) -> None:
        """Should not yield audio attributes if no AUDIO modality present."""
        result = dict(
            _get_token_count_attributes_from_usage_metadata(
                types.GenerateContentResponseUsageMetadata(
                    prompt_tokens_details=[
                        types.ModalityTokenCount(
                            modality=types.MediaModality.TEXT, token_count=100
                        ),
                        types.ModalityTokenCount(
                            modality=types.MediaModality.IMAGE, token_count=50
                        ),
                    ]
                )
            )
        )
        assert SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_AUDIO not in result

    def test_comprehensive_usage_metadata(self) -> None:
        """Should correctly extract all token count attributes from comprehensive metadata."""
        usage = types.GenerateContentResponseUsageMetadata(
            total_token_count=300,
            prompt_token_count=50,
            tool_use_prompt_token_count=30,
            candidates_token_count=100,
            thoughts_token_count=20,
            prompt_tokens_details=[
                types.ModalityTokenCount(modality=types.MediaModality.TEXT, token_count=40),
                types.ModalityTokenCount(modality=types.MediaModality.AUDIO, token_count=10),
            ],
            candidates_tokens_details=[
                types.ModalityTokenCount(modality=types.MediaModality.TEXT, token_count=95),
                types.ModalityTokenCount(modality=types.MediaModality.AUDIO, token_count=5),
            ],
        )
        result = dict(_get_token_count_attributes_from_usage_metadata(usage))
        assert result == {
            SpanAttributes.LLM_TOKEN_COUNT_TOTAL: 300,
            SpanAttributes.LLM_TOKEN_COUNT_PROMPT: 80,  # 50 + 30
            SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_AUDIO: 10,
            SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: 120,  # 100 + 20
            SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_AUDIO: 5,
            SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING: 20,
        }

    def test_zero_token_counts_are_not_yielded(self) -> None:
        """Zero token counts should not be yielded."""
        result = dict(
            _get_token_count_attributes_from_usage_metadata(
                types.GenerateContentResponseUsageMetadata(
                    total_token_count=0, prompt_token_count=0, candidates_token_count=0
                )
            )
        )
        assert result == {}

    def test_missing_token_count_in_modality_details(self) -> None:
        """Should handle missing token_count in modality details."""
        result = dict(
            _get_token_count_attributes_from_usage_metadata(
                types.GenerateContentResponseUsageMetadata(
                    prompt_tokens_details=[
                        types.ModalityTokenCount(
                            modality=types.MediaModality.AUDIO
                        ),  # no token_count
                        types.ModalityTokenCount(
                            modality=types.MediaModality.AUDIO, token_count=10
                        ),
                    ]
                )
            )
        )
        assert result == {SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_AUDIO: 10}

    def test_from_dict_via_model_validate(self) -> None:
        """Should work when caller converts dict via model_validate."""
        usage_dict = {
            "total_token_count": 100,
            "prompt_token_count": 30,
            "candidates_token_count": 70,
        }
        usage = types.GenerateContentResponseUsageMetadata.model_validate(usage_dict)
        result = dict(_get_token_count_attributes_from_usage_metadata(usage))
        assert result == {
            SpanAttributes.LLM_TOKEN_COUNT_TOTAL: 100,
            SpanAttributes.LLM_TOKEN_COUNT_PROMPT: 30,
            SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: 70,
        }
