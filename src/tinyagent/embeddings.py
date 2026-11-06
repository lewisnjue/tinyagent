from tinyagent.http import post_json


class EmbeddingModel:
    """Generate embeddings from an OpenAI-compatible embeddings endpoint."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434/v1",
        timeout: float = 120,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def embed(self, text: str | list[str]) -> list[float] | list[list[float]]:
        """Embed one string or a batch of strings."""
        response = post_json(
            f"{self.base_url}/embeddings",
            {"model": self.model, "input": text},
            {"Content-Type": "application/json"},
            self.timeout,
        )
        vectors = [item["embedding"] for item in response["data"]]
        return vectors[0] if isinstance(text, str) else vectors
