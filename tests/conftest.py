from tinyagent.types import Response


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def generate(self, messages, tools=None):
        self.requests.append(
            {"messages": [dict(message) for message in messages], "tools": tools}
        )
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def response(content="", **kwargs):
    return Response(content=content, **kwargs)
