# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx>=0.28,<1"]
# ///
"""Explore TypeSafe input/output shapes with `uv run raw_http.py`."""

import json
import os

import httpx

EXAMPLES = {
    "Content moderation — independent Noul judgments": {
        "state": (
            'A player wrote "I will find you and hurt you" in our game chat. '
            "That kind of threat is unacceptable. The moderator who ignored my report "
            "is an incompetent idiot. Please investigate."
        ),
        "questions": {
            "makes_threat": {
                "type": "noul",
                "instructions": "Is the author personally threatening someone?",
                "criteria": {
                    "true": "The author expresses their own intent to harm someone.",
                    "false": "The author only quotes, reports, or condemns someone else's threat.",
                },
            },
            "contains_harassment": {
                "type": "noul",
                "instructions": "Does the author direct a personal insult at someone?",
            },
            "quotes_harmful_content": {
                "type": "noul",
                "instructions": "Does the message quote harmful content in order to report it?",
            },
        },
    },
    "Support routing — multiple classifications over a conversation": {
        "state": [
            {"role": "customer", "text": "My upgrade payment went through twice."},
            {"role": "agent", "text": "Does your account show the upgraded plan?"},
            {
                "role": "customer",
                "text": (
                    "No, it still says Free and the export button gives a 500 error. "
                    "I need exports working for a demo today; refund the duplicate charge too."
                ),
            },
        ],
        "questions": {
            "department": {
                "type": "choice",
                "instructions": {
                    "task": "Choose the team that should take ownership of the conversation.",
                    "routing_policy": "Prioritize restoring a blocked workflow over refunds.",
                },
                "criteria": {
                    "billing": {"handles": ["duplicate charges", "refunds", "invoices"]},
                    "technical": {"handles": ["errors", "missing access", "broken workflows"]},
                    "sales": "Plan selection and pricing before purchase.",
                    "other": None,
                },
            },
            "primary_intent": {
                "type": "choice",
                "instructions": "What outcome is most important to the customer right now?",
                "criteria": {
                    "restore_access": "Unblock use of the product.",
                    "get_refund": "Recover money paid.",
                    "cancel_account": "Stop using the product.",
                    "other": "None of these outcomes fits.",
                },
            },
        },
    },
    "Answer quality — scores with different rubric lengths": {
        "state": {
            "question": "Can I return an opened headset after 20 days, and who pays shipping?",
            "reference": {
                "return_window_days": 30,
                "opened_items": "Accepted if undamaged and all accessories are included.",
                "return_shipping": "Customer pays unless the item is defective.",
            },
            "answer": (
                "Yes, opened headsets can be returned within 30 days if undamaged. "
                "We always cover return shipping."
            ),
        },
        "questions": {
            "correctness": {
                "type": "score",
                "instructions": {
                    "task": "Rate the answer's factual accuracy against `reference`.",
                    "focus": ["return window", "opened-item conditions", "shipping payer"],
                },
                "criteria": [
                    {"level": "incorrect", "description": "All material claims are wrong."},
                    {"level": "mixed", "description": "Some claims are right, others wrong."},
                    {"level": "correct", "description": "All stated claims are supported."},
                ],
            },
            "completeness": {
                "type": "score",
                "instructions": [
                    "Rate how fully `answer` addresses `question` using `reference`.",
                    "Include conditions and exceptions needed to act on the answer.",
                ],
                "criteria": [
                    "No useful information.",
                    "Addresses only one part of the question.",
                    "Addresses both parts but omits important conditions.",
                    "Covers both parts and every relevant condition and exception.",
                ],
            },
            "clarity": {
                "type": "score",
                "instructions": "Rate readability independently of accuracy and completeness.",
                "criteria": ["Confusing or difficult to follow.", "Clear and easy to follow."],
            },
        },
    },
    "Refund review — Noul, Choice, and Score together": {
        "state": {
            "order": {
                "days_since_delivery": 12,
                "item": "Wireless headset",
                "condition": "Left speaker stopped working.",
                "accessories_complete": True,
            },
            "policy": {
                "refund_window_days": 30,
                "defective_items": "Eligible for refund or replacement with prepaid shipping.",
            },
            "history": [
                {"days_ago": 3, "event": "Customer reported the defect; no agent response."},
                {"days_ago": 1, "event": "Customer followed up asking for a replacement."},
            ],
            "message": (
                "I need a working headset for accessibility at work tomorrow. "
                "A replacement would be best, but refund me if you cannot ship one today."
            ),
        },
        "questions": {
            "refund_eligible": {
                "type": "noul",
                "instructions": "Does this order qualify for a refund under `policy`?",
            },
            "recommended_action": {
                "type": "choice",
                "instructions": (
                    "Choose the next action using the customer's preference and `policy`. "
                    "Do not assume replacement stock or shipping availability."
                ),
                "criteria": {
                    "check_replacement": "Check whether a replacement can ship today.",
                    "refund": "Issue a refund immediately.",
                    "deny": "Deny the request because it is outside policy.",
                    "request_details": "Ask for information needed to determine eligibility.",
                },
            },
            "urgency": {
                "type": "score",
                "instructions": "How urgently should an agent handle this case?",
                "criteria": [
                    "Routine: no deadline or meaningful disruption.",
                    "Soon: inconvenience with a usable workaround.",
                    "Today: blocked activity or a near-term deadline.",
                    "Immediately: essential accessibility need and an imminent deadline.",
                ],
            },
        },
    },
}


def main() -> None:
    api_key = os.environ.get("TYPESAFE_API_KEY")
    if not api_key:
        raise SystemExit("Set TYPESAFE_API_KEY before running this example.")

    with httpx.Client(
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=60.0,
    ) as client:
        for name, example in EXAMPLES.items():
            request = {"model": "jev-latest", **example}
            print(f"\n=== {name} ===", flush=True)
            print("Request:", flush=True)
            print(json.dumps(request, indent=2, ensure_ascii=False), flush=True)
            try:
                response = client.post("https://api.typesafe.ai/v1/systemone", json=request)
                response.raise_for_status()
            except httpx.HTTPStatusError as error:
                raise SystemExit(
                    f"{name}: HTTP {error.response.status_code}\n{error.response.text}"
                ) from None
            except httpx.RequestError as error:
                raise SystemExit(f"{name}: request failed: {error}") from None
            print("Response:", flush=True)
            print(json.dumps(response.json(), indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
