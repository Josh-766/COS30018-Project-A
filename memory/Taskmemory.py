

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class Observation:
    action: str
    tool_used: str | None
    output: str
    success: bool
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class WorkingMemory:
    goal: str
    history: list[Observation] = field(default_factory=list)

    def add_step(
        self,
        action: str,
        output: str,
        *,
        tool_used: str | None = None,
        success: bool = True,
    ) -> Observation:
        """Call this after the agent does something and sees the result."""
        obs = Observation(action=action, tool_used=tool_used, output=output, success=success)
        self.history.append(obs)
        return obs

    def get_context(self, max_output_chars: int = 300) -> str:
        """Turn the history into plain text for the next prompt.

        truncates long outputs (sandbox logs etc) so we're not burning
        tokens re-sending huge blobs every iteration.
        """
        if not self.history:
            return f"Goal: {self.goal}\nNo actions taken yet."

        lines = [f"Goal: {self.goal}", "Steps so far:"]
        for i, obs in enumerate(self.history, start=1):
            output = obs.output
            if len(output) > max_output_chars:
                output = output[:max_output_chars] + "... (truncated)"
            tool_note = f" [tool: {obs.tool_used}]" if obs.tool_used else ""
            status = "OK" if obs.success else "FAILED"
            lines.append(f"{i}. {obs.action}{tool_note} -> {status}: {output}")
        return "\n".join(lines)

    # evaluation help

    def success_rate(self) -> float:
        if not self.history:
            return 0.0
        successes = sum(1 for obs in self.history if obs.success)
        return successes / len(self.history)

    def tool_usage_count(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for obs in self.history:
            if obs.tool_used:
                counts[obs.tool_used] = counts.get(obs.tool_used, 0) + 1
        return counts

    def step_count(self) -> int:
        return len(self.history)

    def is_stuck(self) -> bool:
        """True if the last two steps are the exact same action+result - a repeat loop."""
        if len(self.history) < 2:
            return False
        a, b = self.history[-2], self.history[-1]
        return a.action == b.action and a.output == b.output

    def reset(self, new_goal: str) -> None:
        self.goal = new_goal
        self.history = []


if __name__ == "__main__":
    # quick manual check, delete later
    wm = WorkingMemory(goal="fix the failing login test")
    wm.add_step("ran pytest", "1 failed: test_login (AssertionError)", tool_used="sandbox", success=False)
    wm.add_step("edited login.py", "saved file", tool_used="editor", success=True)
    wm.add_step("ran pytest", "1 failed: test_login (AssertionError)", tool_used="sandbox", success=False)

    print(wm.get_context())
    print("\nsuccess rate:", wm.success_rate())
    print("tool usage:", wm.tool_usage_count())
    print("stuck?", wm.is_stuck())
