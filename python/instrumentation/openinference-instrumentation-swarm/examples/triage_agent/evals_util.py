from typing import Optional

import instructor
from openai import OpenAI
from pydantic import BaseModel

__client = instructor.from_openai(OpenAI())


class BoolEvalResult(BaseModel):
    value: bool
    reason: Optional[str]


def evaluate_with_llm_bool(instruction, data) -> BoolEvalResult:
    eval_result, _ = __client.chat.completions.create_with_completion(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": data},
        ],
        response_model=BoolEvalResult,
    )
    return eval_result
