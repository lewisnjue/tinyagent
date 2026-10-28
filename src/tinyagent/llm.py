from dataclasses import dataclass
import json
import urllib.request

@dataclass
class Response:
    """ Strucred response from LLM calls."""
    content : str = ""
    reasoning: str | None = None
    tool_call: dict | None = None
    metadata: dict | None = None 


class LLM:
    def __init__(self,
                 model:str,
                 base_url:str = "http://localhost:11434/v1",
                 api_key: str = "no_key",
                 think:bool = False) -> None:
        """ Initialize the LLM with the given model . """
        self.model = model
        self.base_url  = base_ur
        self.api_key = api_key
        self.think  = think
    def generate(
        self,messages:list[dict], tools: list | None = None
    ) -> Response:
        """ Generate a respose from the LLM given a list of messages . """
        body  = {
            "model": self.model,
            "messages": messages,
        }
        # Tools and reasoning
        if tools:
            body["tools"] = tools
        if not self.think:
            body["reasoning_effort"] = "none"
        # POST to OpenAI compatible /chat/completions endpoint
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data = json.dump(body).encode(),
            headers={
                "Content-Type":"application/json",
                "Authorization":f"Bearer {self.api_key}"
            }
        )
        with urllib.request.urlopen(request) as response:
            data = json.loads(response.read())

        message  = data['choices'][0]['message']
        tool_calls  = message.get('tool_calls')
        tool_call = tool_calls[0] if tool_calls else None 
        metadata = {
            "model": data['model'],
            "prompt_tokens": data['usage']['prompt_tokens'],
            "completion_tokens":data['usage']['completion_tokens']
        }
        # forat as Response dataclass 
        
        return Response{
            content = message.get('content'),
            reasoning = message.get("reasoning"),
            tool_call = tool_call,
            metadata = metadata
        }
