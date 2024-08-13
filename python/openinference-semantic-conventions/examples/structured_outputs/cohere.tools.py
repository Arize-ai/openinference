import json
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

import cohere
import vcr

# tool descriptions that the model has access to
tools = [
    {
        "name": "query_daily_sales_report",
        "description": "Connects to a database to retrieve overall sales volumes and sales "
        "information for a given day.",
        "parameter_definitions": {
            "day": {
                "description": "Retrieves sales data for this day, formatted as YYYY-MM-DD.",
                "type": "str",
                "required": True,
            }
        },
    },
    {
        "name": "query_product_catalog",
        "description": "Connects to a a product catalog with information about all the "
        "products being sold, including categories, prices, and stock levels.",
        "parameter_definitions": {
            "category": {
                "description": "Retrieves product information data for all products in "
                "this category.",
                "type": "str",
                "required": True,
            }
        },
    },
]

# preamble containing instructions about the task and the desired style for the output.
preamble = """
## Task & Context
You help people answer their questions and other requests interactively. You will be asked a
very wide array of requests on all kinds of topics. You will be equipped with a wide range of
search engines or similar tools to help you, which you use to research your answer. You should
focus on serving the user's needs as best you can, which will be wide-ranging.

## Style Guide
Unless the user asks for a different style of answer, you should answer in full sentences,
using proper grammar and spelling.
"""

# user request
message = (
    "Can you provide a sales summary for 29th September 2023, and also give me some "
    "details about the products in the 'Electronics' category, for example their prices "
    "and stock levels?"
)

co = cohere.Client()
with ExitStack() as stack:
    stack.enter_context(d := TemporaryDirectory())
    cass = stack.enter_context(
        vcr.use_cassette(
            Path(d.name) / Path(__file__).with_suffix(".yaml").name,
            filter_headers=["authorization"],
            decode_compressed_response=True,
            ignore_localhost=True,
        )
    )
    response = co.chat(
        message=message,
        force_single_step=True,
        tools=tools,
        preamble=preamble,
        model="command-r",
    )
pairs = [
    {
        f"REQUEST-{i}": json.loads(cass.requests[i].body),
        f"RESPONSE-{i}": json.loads(cass.responses[i]["body"]["string"]),
    }
    for i in range(len(cass.requests))
]
with open(Path(__file__).with_suffix(".vcr.json"), "w") as f:
    json.dump(pairs, f, indent=2)
