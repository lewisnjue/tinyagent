from tinyagent.llm import LLM, Response
from tinyagent.memory import Memory
from tinyagent.tools import Tools
from tinyagent.planning import ReAct
from tinyagent.trajectory import Trajectory


class TinyAgent:
    """A minimal, modular, and educational agent framework."""

    def __init__(self, llm: LLM, memory: Memory, tools: Tools, planner: ReAct):
        self.llm = llm
        self.memory = memory
        self.tools = tools
        self.planner = planner
        self.trajectory = Trajectory()
        # Build system prompt with all components
        system_prompt = "You are a helpful assistant.\n\n"
        system_prompt += self.planner.prompt
        system_prompt += self.tools.prompt
        self.memory.add("system", system_prompt)

    def run(self, task: str) -> str:
        """Run the agent on a task."""

        self.memory.add(role="user", content=task)
        self.trajectory.initialize(task)
        for step in range(self.planner.max_steps):
            result = self._step()
            if result is not None:
                return result
 
        return "Max steps reached without completion."

    def _step(self) -> str | None:
        """Perform a single step."""
        response = self.llm.generate(
            self.memory.get_messages(), tools=self.tools.schemas
        )
        self.memory.add(
            "assistant", response.content, tool_call=response.tool_call
        )
 
        # Tool parsing
        response = self.planner.parse(response)
        response = self.tools.parse(response)
 
        # ANSWER: Stopping mechanism
        if self.tools.is_done(response):
            self.trajectory.add(response)
            return response.content
 
        return self._execute_action(response)

    def _execute_action(self, response: Response) -> None:
        """Execute a tool action."""
 
        # ACTION: execute tools
        result = self.tools.execute(response)
 
        # OBSERVATION: add tool results to memory and display
        role, observation = self.tools.observation(result)
        self.memory.add(role, observation)
        self.trajectory.add(response, observation)
 
        return None
