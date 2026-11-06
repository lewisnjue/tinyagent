from tinyagent import LLM, TinyAgent

agent = TinyAgent.native(LLM(model="gemma4:e4b"))
print(agent.run("Explain in one sentence what memory does in an AI agent."))
