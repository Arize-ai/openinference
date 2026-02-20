"""
main.py — mirrors production main.py

Changes from production:
  - arize.otel.register replaced with InMemorySpanExporter + ConsoleSpanExporter
    (optional OTLP to Phoenix if running on localhost:6006)
  - config.yaml loading removed; secrets read from environment variables
  - vcrpy cassette wraps the kickoff() call to record/replay HTTP requests made
    by ScrapeWebsiteTool (requests-based), enabling fast offline re-runs.
    OpenAI API calls use httpx and are NOT intercepted by vcrpy — they always
    go through to the real API.

Run:
  cd examples/flow_crew_context_propagation
  python main.py

  # Pure replay (no network calls for scraping):
  VCR_RECORD_MODE=none python main.py

  # Re-record cassette from scratch:
  VCR_RECORD_MODE=all python main.py
"""

import socket
from pathlib import Path
from typing import Optional

import vcr as vcrpy
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.crewai import CrewAIInstrumentor

# ─── OTel setup (BEFORE crewai imports) ───────────────────────────────────────
memory_exporter = InMemorySpanExporter()
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(memory_exporter))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))


_PHOENIX_HOST = "0.0.0.0"
_PHOENIX_PORT = 6006


def _phoenix_is_running(host: str = _PHOENIX_HOST, port: int = _PHOENIX_PORT) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


_otlp_enabled = False
if _phoenix_is_running():
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter,  # type: ignore[import-not-found]
    )

    otlp_exporter = OTLPSpanExporter(
        endpoint=f"http://{_PHOENIX_HOST}:{_PHOENIX_PORT}/v1/traces"
    )
    tracer_provider.add_span_processor(SimpleSpanProcessor(otlp_exporter))
    _otlp_enabled = True

CrewAIInstrumentor().instrument(tracer_provider=tracer_provider)

# ─── crewai imports (AFTER instrumentation) ───────────────────────────────────
import os  # noqa: E402

from crew import CompanyResearchCrew  # noqa: E402
from crewai.flow import Flow, listen, start  # noqa: E402
from pydantic import BaseModel  # noqa: E402

# ─── VCR setup ────────────────────────────────────────────────────────────────
# vcrpy intercepts requests-library calls (used by ScrapeWebsiteTool).
# OpenAI API calls use httpx and pass through to the real API unchanged.
_CASSETTE_DIR = Path(__file__).parent / "cassettes"
_CASSETTE_DIR.mkdir(exist_ok=True)

_record_mode = os.environ.get("VCR_RECORD_MODE", "once")

_vcr = vcrpy.VCR(
    cassette_library_dir=str(_CASSETTE_DIR),
    record_mode=_record_mode,
    filter_headers=["authorization", "x-api-key", "api-key"],
    match_on=["method", "scheme", "host", "port", "path", "query"],
    decode_compressed_response=True,
    # Don't intercept OTLP exports to the local Phoenix instance.
    ignore_hosts=["0.0.0.0", "localhost", "127.0.0.1"],
)


# ─── Flow definition (mirrors production main.py) ─────────────────────────────
class FlowState(BaseModel):
    lead_data: Optional[dict] = None
    company_research_output: Optional[dict] = None


class DocusignOutreachAiFlow(Flow[FlowState]):
    def __init__(self) -> None:
        super().__init__()
        self.name = "Lead_email_agent"

    @start()
    def get_data(self) -> dict:
        self.state.lead_data["white_paper_email_preface"] = ""
        self.state.lead_data["docusign_solutions"] = (
            "industry_solutions_link.get(industry_classfication_string, None)"
        )
        self.state.lead_data["customer_success_stories"] = (
            "customer_success_stories_link.get(industry_classfication_string) "
            "or customer_success_stories_link.get('Other') or []"
        )
        self.state.lead_data["success_story_link_1"] = "customer_success_stories[0]"
        self.state.lead_data["success_story_link_2"] = "customer_success_stories[1]"
        self.state.lead_data["company"] = self.state.lead_data.get("company", "")
        self.state.lead_data["locale"] = "locale"
        return self.state.lead_data

    @listen(get_data)
    def get_company_research(self, inputs: dict) -> None:
        CompanyResearchCrew().crew().kickoff(inputs=inputs)


# ─── Span verification ─────────────────────────────────────────────────────────
def verify_spans() -> bool:
    """
    Print the span hierarchy and return True if all spans are in one trace.
    """
    spans = memory_exporter.get_finished_spans()

    if not spans:
        print("\n[verify_spans] WARNING: No spans collected.")
        return False

    traces: dict = {}
    for span in spans:
        tid = span.context.trace_id
        traces.setdefault(tid, []).append(span)

    span_by_id = {span.context.span_id: span for span in spans}

    print("\n" + "=" * 70)
    print("SPAN HIERARCHY REPORT")
    print("=" * 70)
    print(f"Total spans   : {len(spans)}")
    print(f"Distinct traces: {len(traces)}")
    print()

    def _tree(span: object, trace_spans: list, indent: int = 0) -> None:
        from opentelemetry.sdk.trace import ReadableSpan

        assert isinstance(span, ReadableSpan)
        sid = format(span.context.span_id, "016x")
        pid = format(span.parent.span_id, "016x") if span.parent else "None (root)"
        kind = (span.attributes or {}).get("openinference.span.kind", "—")
        pad = "  " * indent
        print(f"{pad}{span.name}")
        print(f"{pad}  kind={kind}  id={sid[:12]}  parent={pid[:12]}")
        for child in sorted(
            [s for s in trace_spans if s.parent and s.parent.span_id == span.context.span_id],
            key=lambda s: s.start_time or 0,
        ):
            _tree(child, trace_spans, indent + 1)

    for idx, (tid, tspans) in enumerate(traces.items(), 1):
        print(f"Trace {idx}  (id={format(tid, '032x')[:24]}...)")
        print(f"  Spans: {len(tspans)}")
        print()
        roots = [s for s in tspans if s.parent is None or s.parent.span_id not in span_by_id]
        for r in sorted(roots, key=lambda s: s.start_time or 0):
            _tree(r, tspans, indent=2)
        print()

    # Orphaned TOOL/AGENT/CHAIN/LLM spans whose parent is in a different trace
    orphans = [
        s
        for s in spans
        if s.parent is not None
        and s.parent.span_id not in span_by_id
        and (s.attributes or {}).get("openinference.span.kind", "")
        in ("TOOL", "CHAIN", "AGENT", "LLM")
    ]
    if orphans:
        print("WARNING — orphaned spans (parent in a different trace):")
        for s in orphans:
            kind = (s.attributes or {}).get("openinference.span.kind", "")
            print(f"  [{kind:6s}] {s.name}")
        print()

    print("=" * 70)
    passed = len(traces) == 1
    if passed:
        print("RESULT: PASS — all spans in one trace.")
    else:
        print(
            f"RESULT: FAIL — {len(traces)} separate traces.\n\n"
            "Likely causes:\n"
            "  1. Flow CHAIN separate from Crew CHAIN → asyncio.run() inside\n"
            "     flow.kickoff() creates a new event loop, resetting contextvars.\n"
            "  2. TOOL spans orphaned → ThreadPoolExecutor without context copy.\n"
            "  3. 'Flow Creation' span in its own trace → crewAI built-in telemetry.\n"
            "     Set CREWAI_DISABLE_TELEMETRY=true to suppress it."
        )
    print("=" * 70)
    return passed


# ─── Entrypoint (mirrors production kickoff()) ────────────────────────────────
def kickoff() -> None:
    row_inputs = {
        "title": "Sales Support Manager",
        "company": "ARROW ELECTRONICS ANZ HOLDINGS PTY",
        "country": "Australia",
        "lead_id": "00QUb00000a11BrMAI",
        "segment": "SMB",
        "industry": "Manufacturing",
        "use_case": "ANZ_MM_white_paper_agent",
        "job_level": "Manager-Level",
        "department": "Sales",
        "lead_score": 53,
        "lead_source": "EBOOK_AU_26Q1_IAMforSalesApplicationBrief",
        "sequence_id": 82068,
        "company_size": 112,
        "email_domain": "arrow.com",
        "annual_revenue": 84265253,
        "lead_source_link": (
            "https://img.en25.com/Web/DocuSign/"
            "%7Bd51b9d69-43e5-40c3-a041-a2da00888dc9%7D_"
            "Docusign_IAM_for_Sales_Application_Brief_AU.pdf"
        ),
        "docusign_products": "https://www.docusign.com/products",
        "docusign_services": "https://www.docusign.com/products/platform",
        "outreach_prospect_id": "60691358",
        "lead_source_description": "IAM Sales Application Brief: Maximise Seller Productivity",
        "other_industry_usecases": (
            "SalesContracts,NonDisclosureAgreements,PurchaseAgreements,"
            "EmploymentContracts,ServiceAgreements"
        ),
        "trial_or_white_paper_group": "white_paper",
        "industry_classfication_string": "Manufacturing",
        "lead_owner_outreach_mailbox_id": "30169",
    }

    print("CrewAI Flow+Crew Context Propagation Debug")
    print(f"CREWAI_DISABLE_TELEMETRY={os.environ.get('CREWAI_DISABLE_TELEMETRY', '(not set)')}")
    print(f"VCR_RECORD_MODE={_record_mode}")
    print(f"OTLP to Phoenix: {'yes' if _otlp_enabled else 'no'}")
    print()

    flow = DocusignOutreachAiFlow()

    with _vcr.use_cassette("scrape_requests.yaml"):
        flow.kickoff(inputs={"lead_data": row_inputs})

    verify_spans()


if __name__ == "__main__":
    kickoff()

    if hasattr(tracer_provider, "shutdown"):
        tracer_provider.shutdown()
