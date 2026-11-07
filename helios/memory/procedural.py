"""Procedural Memory: Tool and skill registry for Helios."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from helios.memory.base import MemoryItem


@dataclass
class ToolParameter:
    """Specification of an individual tool argument."""

    name: str
    type: str  # "string", "number", "integer", "boolean", "array", "object"
    description: str
    required: bool = True
    default: Any = None
    enum: list[Any] | None = None


@dataclass
class ToolDefinition:
    """Formal schema and metadata for an approved executable tool."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema (type: object, properties: {...})
    returns: dict[str, Any] = field(default_factory=lambda: {"type": "string"})
    required_parameters: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)
    version: str = "1.0.0"
    author: str = "system"
    is_verified: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    handler: Callable[..., Any] | None = None


@dataclass
class WorkflowStep:
    """Individual execution step within a multi-step verified workflow."""

    step_id: str
    tool_name: str
    description: str
    input_mapping: dict[str, str] = field(default_factory=dict)
    condition: str | None = None
    on_failure: str = "abort"  # "abort" | "retry" | "fallback"


@dataclass
class WorkflowDefinition:
    """Structured, verified execution sequence."""

    name: str
    description: str
    steps: list[WorkflowStep]
    inputs_schema: dict[str, Any] = field(default_factory=dict)
    outputs_schema: dict[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"
    is_verified: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


def validate_parameter_schema(schema: dict[str, Any]) -> None:
    """Validate JSON Schema structure for tool parameters."""
    if not isinstance(schema, dict):
        raise ValueError("Parameter schema must be a dict.")
    if "type" in schema:
        if schema.get("type") != "object":
            raise ValueError("Top-level parameter schema type must be 'object'.")
    elif not (schema and all(isinstance(v, (str, dict)) for v in schema.values())):
        raise ValueError("Top-level parameter schema type must be 'object'.")
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        raise ValueError("'properties' must be a dictionary.")
    valid_types = {
        "string",
        "number",
        "integer",
        "boolean",
        "array",
        "object",
        "null",
    }
    for prop_name, prop_spec in properties.items():
        if not isinstance(prop_spec, dict):
            raise ValueError(f"Property '{prop_name}' must be a dictionary spec.")
        p_type = prop_spec.get("type")
        if p_type not in valid_types:
            raise ValueError(
                f"Property '{prop_name}' has invalid type '{p_type}'. Must be one of {valid_types}."
            )


class ProceduralMemory:
    """Structured skill registry: tools, API schemas, and verified execution workflows."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self._workflows: dict[str, WorkflowDefinition] = {}
        self._capability_index: dict[str, set[str]] = {}

    def register_tool(self, tool: ToolDefinition) -> None:
        """Register tool after schema validation and index its capabilities."""
        if not tool.name or not isinstance(tool.name, str):
            raise ValueError("Tool name must be a non-empty string.")
        validate_parameter_schema(tool.parameters)
        self._tools[tool.name] = tool
        for cap in tool.capabilities:
            cap_key = cap.lower().strip()
            self._capability_index.setdefault(cap_key, set()).add(tool.name)

    def get_tool(self, name: str) -> ToolDefinition | None:
        """Retrieve tool definition by exact name."""
        return self._tools.get(name)

    def find_by_capability(self, capability: str) -> list[ToolDefinition]:
        """Find all registered tools matching a capability tag."""
        names = self._capability_index.get(capability.lower().strip(), set())
        return [self._tools[n] for n in sorted(names)]

    def search_tools(self, query: str, top_k: int = 5) -> list[ToolDefinition]:
        """Substring & capability search for tools relevant to query."""
        q = query.lower()
        matched: list[tuple[int, ToolDefinition]] = []
        for tool in self._tools.values():
            score = 0
            if q in tool.name.lower():
                score += 3
            if any(q in cap.lower() for cap in tool.capabilities):
                score += 2
            if q in tool.description.lower():
                score += 1
            if score > 0:
                matched.append((score, tool))
        matched.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in matched[:top_k]]

    def register_workflow(self, workflow: WorkflowDefinition) -> None:
        """Register verified workflow and validate referenced tools."""
        if not workflow.name:
            raise ValueError("Workflow name cannot be empty.")
        if not workflow.steps:
            raise ValueError("Workflow must contain at least one step.")
        self._workflows[workflow.name] = workflow

    def get_workflow(self, name: str) -> WorkflowDefinition | None:
        """Retrieve workflow definition by name."""
        return self._workflows.get(name)

    def list_tools(self, verified_only: bool = False) -> list[ToolDefinition]:
        """List all tools, optionally filtering for verified skills."""
        tools = list(self._tools.values())
        if verified_only:
            tools = [t for t in tools if t.is_verified]
        return sorted(tools, key=lambda t: t.name)

    def list_workflows(self, verified_only: bool = False) -> list[WorkflowDefinition]:
        """List all workflows, optionally filtering for verified workflows."""
        workflows = list(self._workflows.values())
        if verified_only:
            workflows = [w for w in workflows if w.is_verified]
        return sorted(workflows, key=lambda w: w.name)

    def to_openai_tools(self) -> list[dict[str, Any]]:
        """Export tools to OpenAI / Helios standard function-calling schema."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self.list_tools(verified_only=True)
        ]

    def to_helios_tool_tokens(self) -> str:
        """Export tools formatted with Helios control tokens."""
        blocks: list[str] = []
        for tool in self.list_tools(verified_only=True):
            schema_str = json.dumps(tool.parameters)
            block = (
                f"<|call:tool|>{tool.name}\n"
                f"Description: {tool.description}\n"
                f"Parameters: {schema_str}\n"
                f"<|return:tool|>"
            )
            blocks.append(block)
        return "\n".join(blocks)

    def to_memory_items(self) -> list[MemoryItem]:
        """Convert all registered tools and workflows into MemoryItem representation."""
        items: list[MemoryItem] = []
        for tool in self._tools.values():
            content = (
                f"Tool: {tool.name}\nDescription: {tool.description}\nParameters: {tool.parameters}"
            )
            items.append(
                MemoryItem(
                    id=f"proc_tool_{tool.name}",
                    content=content,
                    tier="procedural",
                    metadata={
                        "name": tool.name,
                        "capabilities": tool.capabilities,
                        "type": "tool",
                    },
                    score=1.0,
                )
            )
        for wf in self._workflows.values():
            steps_desc = "; ".join(f"{s.step_id}: {s.tool_name}" for s in wf.steps)
            content = f"Workflow: {wf.name}\nDescription: {wf.description}\nSteps: {steps_desc}"
            items.append(
                MemoryItem(
                    id=f"proc_wf_{wf.name}",
                    content=content,
                    tier="procedural",
                    metadata={"name": wf.name, "type": "workflow"},
                    score=1.0,
                )
            )
        return items
