import json
import urllib.error
import urllib.request
from typing import Any


class LLMError(RuntimeError):
    """Raised when an OpenAI-compatible endpoint cannot complete a request."""


def post_json(
    url: str, body: dict, headers: dict[str, str], timeout: float = 120
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read())
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise LLMError(f"HTTP {error.code} from {url}: {details}") from error
    except urllib.error.URLError as error:
        raise LLMError(
            f"Could not reach {url}. Is the service running? {error.reason}"
        ) from error
    except json.JSONDecodeError as error:
        raise LLMError(f"Endpoint {url} returned invalid JSON: {error}") from error
    if not isinstance(data, dict):
        raise LLMError(f"Expected a JSON object from {url}.")
    return data
