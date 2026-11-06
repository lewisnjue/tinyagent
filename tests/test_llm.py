import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from tinyagent.http import LLMError
from tinyagent.llm import LLM


class ChatHandler(BaseHTTPRequestHandler):
    status = 200
    payload = {}
    request_body = {}

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        ChatHandler.request_body = json.loads(self.rfile.read(length))
        body = json.dumps(self.payload).encode()
        self.send_response(self.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


@pytest.fixture
def endpoint():
    server = ThreadingHTTPServer(("127.0.0.1", 0), ChatHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_response_handles_missing_usage_and_reasoning(endpoint):
    ChatHandler.status = 200
    ChatHandler.payload = {
        "model": "test",
        "choices": [
            {
                "message": {
                    "content": None,
                    "reasoning_content": "thinking",
                }
            }
        ],
    }
    response = LLM("test", base_url=endpoint).generate([])
    assert response.content == ""
    assert response.reasoning == "thinking"
    assert response.metadata["prompt_tokens"] == 0


def test_empty_native_tools_are_omitted_from_request(endpoint):
    from tinyagent.tools import NativeTools

    ChatHandler.status = 200
    ChatHandler.payload = {
        "choices": [{"message": {"content": "answer"}}],
    }
    LLM("test", base_url=endpoint).generate([], tools=NativeTools().schemas)
    assert "tools" not in ChatHandler.request_body


def test_http_error_includes_provider_body(endpoint):
    ChatHandler.status = 400
    ChatHandler.payload = {"error": "bad request details"}
    with pytest.raises(LLMError, match="bad request details"):
        LLM("test", base_url=endpoint).generate([])
