from fastapi.testclient import TestClient

from optimizer import serve
from optimizer.backends import MockLLM


def test_browser_ui_and_model_endpoint_are_available(monkeypatch):
    monkeypatch.setattr(serve, "BACKEND", MockLLM())
    client = TestClient(serve.app)
    home = client.get("/")
    assert home.status_code == 200
    assert "Prompt Optimizer" in home.text
    assert client.get("/static/styles.css").status_code == 200
    assert client.get("/static/polish.css").status_code == 200
    answer = client.post("/query", json={"text": "Extract a JSON record from this note"})
    assert answer.status_code == 200
    assert answer.json()["trace_id"].startswith("trace_")
    history = client.get("/traces")
    assert history.status_code == 200
    assert "traces" in history.json()
