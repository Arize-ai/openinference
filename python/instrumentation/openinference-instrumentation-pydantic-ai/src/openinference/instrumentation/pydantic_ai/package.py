# The supported Pydantic AI V1 and V2 ranges require different genai-prices
# constraints. Package requirements cannot express that conditional relationship;
# the resolver-safe pairs live in the instruments extras and CI requirement files.
_instruments = ("pydantic-ai >= 1.107.2",)
_supports_metrics = False
