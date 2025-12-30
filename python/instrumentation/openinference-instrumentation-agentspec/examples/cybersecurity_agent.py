# This demo is inspired by the agent built at:
# https://medium.com/oracledevs/tutorial-evolving-cybersecurity-with-open-agent-spec-and-wayflow-f7ecd3b6e7df

# It requires the latest version of ``wayflowcore`` installed. Please run `pip install "wayflowcore>=26.1.0"`.
# Otherwise, install the latest version from source with `pip install git+https://github.com/oracle/wayflow.git#subdirectory=wayflowcore`.

import random
from typing import List

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import Resource
from pyagentspec.agent import Agent
from pyagentspec.flows.edges import ControlFlowEdge, DataFlowEdge
from pyagentspec.flows.flow import Flow
from pyagentspec.flows.node import Node
from pyagentspec.flows.nodes import (
    AgentNode,
    BranchingNode,
    EndNode,
    InputMessageNode,
    LlmNode,
    OutputMessageNode,
    StartNode,
    ToolNode,
)
from pyagentspec.llms import OpenAiConfig
from pyagentspec.property import DictProperty, ListProperty, StringProperty
from pyagentspec.tools import ServerTool
from pyagentspec.tracing.trace import Trace
from wayflowcore.agentspec import AgentSpecLoader
from wayflowcore.agentspec.tracing import AgentSpecEventListener
from wayflowcore.events.eventlistener import register_event_listeners
from wayflowcore.executors.executionstatus import FinishedStatus, UserMessageRequestStatus

from openinference.instrumentation.agentspec import OpenInferenceSpanProcessor
from openinference.semconv.resource import ResourceAttributes


def retrieve_tenancy_graph(tenancy_id: str) -> dict:
    # In this tool, a user could theoretically query a data source
    # to obtain the tenancy resources, or even scan it in real time through API.
    if tenancy_id == "vulnerable_tenancy":
        return {
            "vertices": [{"id": "RESOURCE_ID", "fields": {"field": "METADATA"}}],  # string, not set
            "edges": [{"source": "RESOURCE_ID", "target": "DESTINATION_ID"}],
        }
    return {}


def find_networking_vulnerabilities(tenancy_graph: dict) -> List[dict]:
    # Starting from networking resources, you could scan their IP
    # with nmap, Shodan or others network scanning tool, to detect
    # exposed ports and services.
    # In this example we return a mock finding.
    return [
        {
            "resource": "nsg1",
            "tenancy": "vulnerable_tenancy",
            "summary": "",
            "verified": False,
            "vuln_type": "exposed_port",
            "metadata": {
                "IP": "100.100.100.100",
                "PORT": "22",
            },
            "severity": "Not computed",
            "additional_details": "Not computed",
        }
    ]


def find_sensitive_files(tenancy_graph: dict, findings: List[dict]) -> List[dict]:
    # You could access public or misconfigured object storage instances
    # in the tenancy and then scan their contents for sensitive files.
    # The same thing could be done for exposed NFS servers
    # offering NFS exports that are mountable outside the tenancy.
    # We return a mock finding for each of the two cases.
    findings = list(findings or [])
    findings.extend(
        [
            {
                "resource": "os1",
                "tenancy": "vulnerable_tenancy",
                "summary": "",
                "verified": False,
                "vuln_type": "sensitive_object_storage_file",
                "metadata": {
                    "OS_NAMESPACE": "some_namespace",
                    "OS_NAME": "some_os",
                    "OBJECT": "/home/user/id_rsa",
                },
                "severity": "Not computed",
                "additional_details": "Not computed",
            },
            {
                "resource": "nfs1",
                "tenancy": "vulnerable_tenancy",
                "summary": "",
                "verified": False,
                "vuln_type": "sensitive_nfs_file",
                "metadata": {
                    "NFS_SERVER_IP": "100.100.100.101",
                    "NFS_EXPORT_PATH": "/nfs",
                    "OBJECT": "/nfs/tmp/user/id_rsa",
                },
                "severity": "Not computed",
                "additional_details": "Not computed",
            },
        ]
    )
    return findings


def evaluate_exploits(tenancy_graph: dict, findings: List[dict]) -> List[dict]:
    # Depending on the type of findings, you can try to exploit it safely
    # to later evaluate severity and blast radius.
    # For example, you could try to authenticate as an user if you have their data
    # or trying SSH into an instance, or exploiting a CVE scanned through Shodan, etc.
    # Here we just assigned a random verification status to the findings.
    # If True, it means the exploit was successfully executed
    for f in findings:
        print(f"Exploiting {f}")
        f["verified"] = random.choice([True, False])
    return findings


def has_findings(findings: List[dict]) -> str:
    if len(findings) > 0:
        print(f"Findings detected: {findings}.")
        return "yes"
    return "no"


def find_lateral_movement_from_user(tenancy_graph: dict, finding: dict) -> dict:
    # You could evaluate what an exploited user could do
    # deriving this info from its permissions and groups
    # This example identifies an exploited admin user
    # that can also access databases in another tenancy
    finding["additional_details"] = {
        "compromised_user": "vulnerable_user",
        "policies": {
            "vulnerable_tenancy": ["allow vulnerable_user to manage all resources"],
            "other_tenancy": ["allow vulnerable_user to use databases"],
        },
        "groups": {
            "vulnerable_tenancy": ["admins"],
            "other_tenancy": ["db_admins"],
        },
    }
    return finding


def find_networking_lateral_movement(tenancy_graph: dict, finding: dict) -> dict:
    # You could run path analysis to understand where an attacker
    # could reach from a compromised instance.
    # In this example, several ports on other instances are reachable
    finding = dict(finding or {})
    finding["additional_details"] = {
        "reachable_ip_port_pairs": [
            "100.100.100.101/22",
            "100.100.100.102/6443",
            "100.100.100.103/2048",
        ]
    }
    return finding


def report_findings(findings: List[dict]) -> bool:
    for f in findings or []:
        print(f"=== Sending email for finding {f.get('resource')} ===")
        print(f"Finding: {f}")
        print("=" * 40)
    return True


# Agent Spec Flow definition


def create_flow() -> Flow:
    def create_control_flow_edge(
        from_node: Node, to_node: Node, from_branch: str | None = None
    ) -> ControlFlowEdge:
        return ControlFlowEdge(
            name=f"{from_node.name}_{to_node.name}_{from_branch}",
            from_node=from_node,
            to_node=to_node,
            from_branch=from_branch,
        )

    def create_data_flow_edge(
        source_node: Node, destination_node: Node, property_name: str
    ) -> DataFlowEdge:
        return DataFlowEdge(
            name=f"{source_node.name}_{destination_node.name}_{property_name}",
            source_node=source_node,
            source_output=property_name,
            destination_node=destination_node,
            destination_input=property_name,
        )

    data_flow_edges: list[DataFlowEdge] = []
    control_flow_edges: list[ControlFlowEdge] = []
    flow_nodes: list[Node] = []

    # Defining the needed AgentSpec property objects
    # The tenancy ID input by the user
    tenancy_id_property = StringProperty(title="tenancy_id", description="The tenancy identifier.")
    # The tenancy graph containing the resources from which the findings can be computed
    tenancy_graph_property = DictProperty(
        title="tenancy_graph",
        description="Tenancy of the graph resources",
        value_type=StringProperty(),
    )
    # The list of findings
    findings_property = ListProperty(
        title="findings",
        description="List of findings",
        item_type=DictProperty(title="finding", value_type=StringProperty(title="field")),
    )
    # Whether findings were detected
    has_findings_property = StringProperty(
        title="has_findings", description="Whether detection yielded findings."
    )

    retrieve_tenancy_graph_tool = ServerTool(
        name="retrieve_tenancy_graph",
        description="Tool to retrieve tenancy graph",
        inputs=[tenancy_id_property],
        outputs=[tenancy_graph_property],
    )

    find_networking_vulnerabilities_tool = ServerTool(
        name="find_networking_vulnerabilities",
        description="Tool to find networking vulnerabilities",
        inputs=[tenancy_graph_property],
        outputs=[findings_property],
    )

    find_sensitive_files_tool = ServerTool(
        name="find_sensitive_files",
        description="Tool to find sensitive files in NFS and OS",
        inputs=[tenancy_graph_property, findings_property],
        outputs=[findings_property],
    )

    evaluate_exploits_tool = ServerTool(
        name="evaluate_exploits",
        description="Tool to safely try exploits",
        inputs=[tenancy_graph_property, findings_property],
        outputs=[findings_property],
    )

    has_findings_tool = ServerTool(
        name="has_findings",
        description="Tool to evaluate whether there are findings",
        inputs=[findings_property],
        outputs=[has_findings_property],
    )

    start_node = StartNode(name="start_node")
    exit_node = EndNode(name="exit_node")

    flow_nodes.extend([start_node, exit_node])

    presentation_message_node = OutputMessageNode(
        name="presentation_message",
        message="Hi, I am the CyberSecurity Assistant. Please insert the ID of the Tenancy you would like to investigate.",
    )

    get_tenancy_id_node = InputMessageNode(
        name="get_tenancy_id",
        outputs=[tenancy_id_property],
    )

    flow_nodes.extend([presentation_message_node, get_tenancy_id_node])

    control_flow_edges.extend(
        [
            # From start to opening message
            create_control_flow_edge(start_node, presentation_message_node),
            # From opening message to user input
            create_control_flow_edge(presentation_message_node, get_tenancy_id_node),
        ]
    )

    def create_tool_node(tool: ServerTool) -> ToolNode:
        return ToolNode(name=f"{tool.name} Node", description=tool.description, tool=tool)

    # Wrapping the tools
    retrieve_tenancy_graph_tool_node = create_tool_node(retrieve_tenancy_graph_tool)
    find_networking_vulnerabilities_tool_node = create_tool_node(
        find_networking_vulnerabilities_tool
    )
    find_sensitive_files_tool_node = create_tool_node(find_sensitive_files_tool)
    evaluate_exploits_tool_node = create_tool_node(evaluate_exploits_tool)

    flow_nodes.extend(
        [
            retrieve_tenancy_graph_tool_node,
            find_networking_vulnerabilities_tool_node,
            find_sensitive_files_tool_node,
            evaluate_exploits_tool_node,
        ]
    )

    control_flow_edges.extend(
        [
            # From user input to first tool
            create_control_flow_edge(get_tenancy_id_node, retrieve_tenancy_graph_tool_node),
            # Cascading tool calls
            create_control_flow_edge(
                retrieve_tenancy_graph_tool_node, find_networking_vulnerabilities_tool_node
            ),
            create_control_flow_edge(
                find_networking_vulnerabilities_tool_node, find_sensitive_files_tool_node
            ),
            create_control_flow_edge(find_sensitive_files_tool_node, evaluate_exploits_tool_node),
        ]
    )

    # Inputs and outputs will need to correspond to what is specified in the individual tools, and thus also to what the MCP server tools need
    data_flow_edges.extend(
        [
            # Need the tenancy ID to retrieve the tenancy graph
            create_data_flow_edge(
                get_tenancy_id_node, retrieve_tenancy_graph_tool_node, "tenancy_id"
            ),
            # Need the tenancy graph to compute the first findings
            create_data_flow_edge(
                retrieve_tenancy_graph_tool_node,
                find_networking_vulnerabilities_tool_node,
                "tenancy_graph",
            ),
            # Second tool still needs the tenancy graph and enriches existing list of findings
            create_data_flow_edge(
                retrieve_tenancy_graph_tool_node, find_sensitive_files_tool_node, "tenancy_graph"
            ),
            create_data_flow_edge(
                find_networking_vulnerabilities_tool_node,
                find_sensitive_files_tool_node,
                "findings",
            ),
            # Last tool evaluates exploits on computed findings and still needs tenancy graph
            create_data_flow_edge(
                retrieve_tenancy_graph_tool_node, evaluate_exploits_tool_node, "tenancy_graph"
            ),
            create_data_flow_edge(
                find_sensitive_files_tool_node, evaluate_exploits_tool_node, "findings"
            ),
        ]
    )

    has_findings_tool_node = create_tool_node(has_findings_tool)

    # Maps the 'yes' outcome to the next node (the output message)
    # or goes back to tenancy ID request by default
    has_findings_branching_node = BranchingNode(
        name="has_findings_branch",
        description="Exit the flow if the are no detected findings",
        mapping={
            "yes": "has_findings",
        },
        inputs=[has_findings_property],
    )

    no_findings_output_node = OutputMessageNode(
        name="no_findings_message",
        message="There were no findings computed. Please input another tenancy ID.",
    )

    flow_nodes.extend(
        [has_findings_tool_node, has_findings_branching_node, no_findings_output_node]
    )

    control_flow_edges.extend(
        [
            # From last tool to branching node
            create_control_flow_edge(evaluate_exploits_tool_node, has_findings_tool_node),
            create_control_flow_edge(has_findings_tool_node, has_findings_branching_node),
            # Go to no findings message by default (will specify alternative behavior later)
            create_control_flow_edge(
                has_findings_branching_node, no_findings_output_node, from_branch="default"
            ),
            # From message to beginning
            create_control_flow_edge(no_findings_output_node, get_tenancy_id_node),
        ]
    )

    # Fetches to see if there are findings
    data_flow_edges.extend(
        [
            create_data_flow_edge(evaluate_exploits_tool_node, has_findings_tool_node, "findings"),
            create_data_flow_edge(
                has_findings_tool_node, has_findings_branching_node, "has_findings"
            ),
        ]
    )

    # We will use a OpenAI GPT-5 mini for this example
    llm_config = OpenAiConfig(name="openai-gpt-5-mini", model_id="gpt-5-mini")

    # In the prompt, the {{<input>}} token is used to pass the inputs
    # and is mandatory if the Node specifies inputs
    SUMMARIZATION_INSTRUCTIONS = """
    You are a CyberSecurity expert tasked to summarize security vulnerability findings.
    You will receive a list of findings in the format:
    - resource: <affected_resource_id>
    - tenancy: <tenancy_id>
    - summary: <summary, starts empty>
    - verified: <boolean describing if the finding yielded a verified exploit>
    - vuln_type: <'exposed_port', 'sensitive_object_storage_file', 'sensitive_nfs_file'>
    - metadata: <added by tools>
    - additional_details: <added by tools>
    - severity: <ignore for now>

    Here's the list of findings:
    {{findings}}

    Populate the summary field with a description of why the findings is problematic and what an attacker could do with it. Use less than 100 for each summary.
    Also, return the updated findings.
    """

    # Remember to specify inputs ad outputs
    summarization_node = LlmNode(
        name="summarize_findings_node",
        llm_config=llm_config,
        prompt_template=SUMMARIZATION_INSTRUCTIONS,
        inputs=[findings_property],
        outputs=[findings_property],
    )

    # Notify user that next step is triaging
    going_to_triage_output_node = OutputMessageNode(
        name="going_to_triage_message",
        message="Moving to triaging Agent. Please wait for triaging to finish.",
    )

    flow_nodes.extend([summarization_node, going_to_triage_output_node])

    control_flow_edges.extend(
        [
            # summarize if there are findings
            create_control_flow_edge(
                has_findings_branching_node, summarization_node, from_branch="has_findings"
            ),
            # when finished, notify the user that triaging is next
            create_control_flow_edge(summarization_node, going_to_triage_output_node),
        ]
    )

    # Feed findings to summarization node
    data_flow_edges.append(
        create_data_flow_edge(evaluate_exploits_tool_node, summarization_node, "findings")
    )

    find_lateral_movement_from_user = ServerTool(
        name="find_lateral_movement_from_user",
        description="Tool to evaluate where a compromised user has rights",
        inputs=[tenancy_graph_property, findings_property],
        outputs=[findings_property],
    )

    find_networking_lateral_movement = ServerTool(
        name="find_networking_lateral_movement",
        description="Tool to evaluate where an attacker could reach from a compromised IP",
        inputs=[tenancy_graph_property, findings_property],
        outputs=[findings_property],
    )

    TRIAGING_INSTRUCTIONS = """
    You are an agent designed to triaging findings of security vulnerabilities within a Cloud tenancy.
    Given a list of findings and a tenancy graph, you have to evaluate the possibilities of moving laterally from the findings entrypoints and their blast radius.

    Here's the list of findings:
    {{findings}}

    Here's the tenancy graph:
    {{tenancy_graph}}

    1) Your workflow (starting from the list of findings and the tenancy graph)
    - For each finding where Verified is True:
        - If finding is of type 'sensitive_object_storage_file' OR 'sensitive_nfs_file':
            1) Call find_lateral_movement_from_user(finding_metadata)
            2) Extract additional_details
        - if finding is of type 'exposed_port':
            1) Call find_networking_lateral_movement(finding_metadata)
            2) Extract additional_details

    - For each finding where Verified is False:
        - Assign LOW Severity
        - Assign 'Finding could not be exploited' as Severity_Explanation

    2) For the output, for each finding use the data in the "metadata" and "additional_details" fields to:
        - summarize the blast radius and lateral movement possibilities for the finding in less than 100 words
        - assign a severity in the severity field, either, 'LOW', 'MEDIUM' or 'HIGH'
        - mind that if a finding is verified, it is likely to have a higher severity

    Then output the updated findings list with a small summary about the issues of the tenancy.
    """.strip()

    triaging_agent = Agent(
        name="Triaging_Agent",
        description="Agent equipped with tools to assist with computing blast radius and lateral movement possibilities from Cloud tenancy findings",
        llm_config=llm_config,
        tools=[find_lateral_movement_from_user, find_networking_lateral_movement],
        system_prompt=TRIAGING_INSTRUCTIONS,
        inputs=[findings_property, tenancy_graph_property],
        outputs=[findings_property],
    )

    triaging_agent_node = AgentNode(name="triaging_agent_node", agent=triaging_agent)

    # Tell the user to input 'yes' for reporting step
    request_confirmation_message_node = OutputMessageNode(
        name="request_confirmation_message",
        message="Would you like to report the findings to their respective tenancy owner via mail? Enter 'yes' if that is the case, otherwise Flow will terminate.",
    )

    flow_nodes.extend([triaging_agent_node, request_confirmation_message_node])

    control_flow_edges.extend(
        [
            # From previous message to the triaging agent
            create_control_flow_edge(going_to_triage_output_node, triaging_agent_node),
            create_control_flow_edge(triaging_agent_node, request_confirmation_message_node),
        ]
    )

    data_flow_edges.extend(
        [
            # Triaging agent needs tenancy graph and findings
            create_data_flow_edge(
                retrieve_tenancy_graph_tool_node, triaging_agent_node, "tenancy_graph"
            ),
            create_data_flow_edge(summarization_node, triaging_agent_node, "findings"),
        ]
    )

    # A simple string property containing the user's confirmation
    confirmation_property = StringProperty(
        title="confirmation", description="Confirmation for reporting."
    )

    confirmation_input_node = InputMessageNode(
        name="get_confirmation",
        outputs=[confirmation_property],
    )

    reporting_branching_node = BranchingNode(
        name="reporting_decision",
        description="Report findings if the users decides to do so",
        mapping={
            "yes": "reporting",
        },
        inputs=[confirmation_property],
    )

    reporting_tool = ServerTool(
        name="report_findings",
        description="Finds the owner of a finding and reports it.",
        inputs=[findings_property],
    )

    reporting_node = ToolNode(
        name="reporting_node",
        description="Sends mail to the finding's tenancy owner.",
        tool=reporting_tool,
    )

    flow_nodes.extend([confirmation_input_node, reporting_branching_node, reporting_node])

    control_flow_edges.extend(
        [
            create_control_flow_edge(request_confirmation_message_node, confirmation_input_node),
            create_control_flow_edge(confirmation_input_node, reporting_branching_node),
            # If the user writes 'yes', move to the reporting tool node
            create_control_flow_edge(
                reporting_branching_node, reporting_node, from_branch="reporting"
            ),
            # Otherwise exit
            create_control_flow_edge(reporting_branching_node, exit_node, from_branch="default"),
            # Exit after reporting
            create_control_flow_edge(reporting_node, exit_node),
        ]
    )

    data_flow_edges.extend(
        [
            # Pass confirmation to branching
            create_data_flow_edge(
                confirmation_input_node, reporting_branching_node, "confirmation"
            ),
            # Pass findings to the reporting tool node
            create_data_flow_edge(triaging_agent_node, reporting_node, "findings"),
        ]
    )

    return Flow(
        name="CyberSec_Assistant_Flow",
        start_node=start_node,
        nodes=flow_nodes,
        control_flow_connections=control_flow_edges,
        data_flow_connections=data_flow_edges,
    )


# Convert the assistant and run it with OpenInference tracing enabled

tool_registry = {
    "retrieve_tenancy_graph": retrieve_tenancy_graph,
    "find_networking_vulnerabilities": find_networking_vulnerabilities,
    "find_sensitive_files": find_sensitive_files,
    "evaluate_exploits": evaluate_exploits,
    "find_lateral_movement_from_user": find_lateral_movement_from_user,
    "has_findings": has_findings,
    "report_findings": report_findings,
    "find_networking_lateral_movement": find_networking_lateral_movement,
}
flow = create_flow()
wayflow_assistant = AgentSpecLoader(tool_registry=tool_registry).load_component(flow)


def conversation_loop(assistant):
    conversation = assistant.start_conversation()
    while True:
        status = conversation.execute()
        if isinstance(status, FinishedStatus):
            break
        assistant_reply = conversation.get_last_message()
        if assistant_reply is not None:
            print("\nAssistant >>>", assistant_reply.content)
        if isinstance(status, UserMessageRequestStatus):
            user_input = input("\nUser >>> ")
            conversation.append_user_message(user_input)


span_processor = OpenInferenceSpanProcessor(
    span_exporter=OTLPSpanExporter("http://127.0.0.1:6006/v1/traces"),
    resource=Resource({ResourceAttributes.PROJECT_NAME: "cybersecurity-agent-trace"}),
    mask_sensitive_information=False,
)

with Trace(span_processors=[span_processor]):
    with register_event_listeners([AgentSpecEventListener()]):
        # Give "vulnerable_tenancy" as input to test it
        conversation_loop(wayflow_assistant)
