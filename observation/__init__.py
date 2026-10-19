"""Observation: watches what the agents do, stops them going in circles, and
logs everything so we can measure the system.

All the code is in observation.py. This file just lets other code write
`from observation import ...` instead of `from observation.observation import ...`.
"""

from .observation import (
    Observation,
    ObservedGraph,
    ObservedSandbox,
    Tracer,
    classify_output,
    create_observed_coder_graph,
    create_observed_orchestrator,
    failure_streak,
    install,
    observe,
    read_trace,
    summarise_output,
    summarise_traces,
    uninstall,
)

__all__ = [
    "Observation",
    "ObservedGraph",
    "ObservedSandbox",
    "Tracer",
    "classify_output",
    "create_observed_coder_graph",
    "create_observed_orchestrator",
    "failure_streak",
    "install",
    "observe",
    "read_trace",
    "summarise_output",
    "summarise_traces",
    "uninstall",
]
