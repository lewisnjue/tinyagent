"""Agent loop order: planner.parse -> tools.parse -> tools.is_done."""

from tinyagent.llm import LLM
from tinyagent.memory import Memory
from tinyagent.planning import NativeReAct, ReAct
from tinyagent.tools import NativeTools, Tools
from tinyagent.trajectory import Trajectory
from tinyagent.types import Response


class TinyAgent:
    """A small, modular, and educational single-agent framework."""

    def __init__(
        self,
        llm: LLM,
        memory: Memory,
        tools: Tools,
        planner: ReAct,
        max_steps: int = 10,
        verbose: bool = False,
    ) -> None:
        if isinstance(tools, NativeTools) != isinstance(planner, NativeReAct):
            raise ValueError("Use NativeTools with NativeReAct, or Tools with ReAct.")
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self.llm = llm
        self.memory = memory
        self.tools = tools
        self.planner = planner
        self.max_steps = max_steps
        self.verbose = verbose
        self.trajectory = Trajectory()

        system_prompt = "\n\n".join(
            part
            for part in (
                "You are a helpful assistant.",
                self.planner.prompt,
                self.tools.prompt,
            )
            if part
        )
        self.memory.add("system", system_prompt)

    @classmethod
    def native(
        cls,
        llm: LLM,
        memory: Memory | None = None,
        tools: NativeTools | None = None,
        max_steps: int = 10,
        verbose: bool = False,
    ) -> "TinyAgent":
        return cls(
            llm,
            memory or Memory(),
            tools or NativeTools(),
            NativeReAct(),
            max_steps=max_steps,
            verbose=verbose,
        )

    @classmethod
    def text(
        cls,
        llm: LLM,
        memory: Memory | None = None,
        tools: Tools | None = None,
        max_steps: int = 10,
        verbose: bool = False,
    ) -> "TinyAgent":
        return cls(
            llm,
            memory or Memory(),
            tools or Tools(),
            ReAct(),
            max_steps=max_steps,
            verbose=verbose,
        )

    def run(self, task: str) -> str:
        """Run a task; subsequent calls continue the same conversation."""
        self.memory.add("user", task)
        self.trajectory.initialize(task)
        last_response = Response()
        for _ in range(self.max_steps):
            result, last_response = self._step()
            if result is not None:
                self.memory.end_turn()
                return result

        self.trajectory.add(
            Response(content=last_response.content, reasoning=last_response.reasoning),
            stopped="max_steps",
        )
        self.memory.end_turn()
        return last_response.content or ""

    def _step(self) -> tuple[str | None, Response]:
        response = self.llm.generate(
            self.memory.get_messages(), tools=self.tools.schemas
        )
        last_response = response
        # Save provider-native tool_calls before parsing changes their shape.
        self.memory.add(
            "assistant",
            response.content,
            tool_calls=response.tool_calls
            if isinstance(self.tools, NativeTools)
            else None,
        )
        response = self.planner.parse(response)
        response = self.tools.parse(response)

        if response.parse_error:
            self._show("OBSERVATION", response.parse_error)
            role, content = self.tools.observation(response.parse_error)
            self.memory.add(role, content)
            self.trajectory.add(response, content)
            return None, response

        if self.tools.is_done(response):
            if response.reasoning:
                self._show("THOUGHT", response.reasoning)
            if response.tool_call:
                self._show("ACTION", response.tool_call)
            self._show("ANSWER", response.content)
            self.trajectory.add(response)
            return response.content or "", response

        actions = response.tool_calls
        for action_index, action in enumerate(actions):
            self._show("THOUGHT", response.reasoning or "")
            self._show("ACTION", action)
            if action.get("error"):
                result = action["error"]
            else:
                result = self.tools.execute(Response(tool_call=action))
            if isinstance(self.tools, NativeTools):
                message = self.tools.observation(result, action.get("id"))
                self.memory.add_message(message)
                observation = message["content"]
            else:
                role, observation = self.tools.observation(result)
                self.memory.add(role, observation)
            self._show("OBSERVATION", observation)
            self.trajectory.add(
                Response(
                    content=response.content,
                    reasoning=response.reasoning,
                    tool_call=action,
                    metadata=response.metadata if action_index == 0 else None,
                ),
                observation,
            )
        return None, last_response

    def _show(self, label: str, value: object) -> None:
        if self.verbose:
            print(f"{label}: {value}")

    def reset(self) -> None:
        """Clear conversational state and recorded runs, keeping system prompts."""
        self.memory.reset()
        self.trajectory = Trajectory()
