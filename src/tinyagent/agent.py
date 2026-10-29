from .llm import LLM, Response
from .tools import Tools
from .trajectory import Trajectory
from .memory import Memory


class TinyAgent:
    """A minimal, modular, and educational agent framework."""

    def __init__(self, llm: LLM, memory: Memory, tools: Tools):
        self.llm = llm
        self.memory = memory
        self.tools = tools
        self.planner = None  # Chapter 6: Add Planning
        self.trajectory = Trajectory()
        system_prompt = "You are a helpful assistant.\n\n"
        system_prompt += self.tools.prompt
        self.memory.add("system", system_prompt)

    def run(self, task: str) -> str:
        """Run the agent on a task."""

        self.memory.add(role="user", content=task)
        self.trajectory.initialize(task)
        return self._step()

    def _step(self) -> str:
        """Perform a single step."""
        # THOUGHT: Generate response and add to memory
        response = self.llm.generate(
            self.memory.get_messages(), tools=self.tools.schemas
        )
        self.memory.add("assistant", response.content, tool_call=response.tool_call)

        # Tool parsing
        response = self.tools.parse(response)

        # ANSWER: Stopping mechanism
        if self.tools.is_done(response):
            self.trajectory.add(response)
            return response.content

        return self._execute_action(response)

    def _execute_action(self, response: Response):
        # ACTION: execute tools
        result = self.tools.execute(response)

        # OBSERVATION: add tool results to memory and display
        role, observation = self.tools.observation(result)
        self.memory.add(role, observation)
        self.trajectory.add(response, observation)
        return observation
