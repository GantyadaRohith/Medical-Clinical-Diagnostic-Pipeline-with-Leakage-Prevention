import os


def _call_resolve_api_url(monkeypatch):
    import importlib

    # Simulate a deployed Streamlit app trying to call its own hostname
    monkeypatch.setenv("API_URL", "https://medical-clinical-diagnostic-pipeline.onrender.com")
    monkeypatch.delenv("BACKEND_API_URL", raising=False)

    module = importlib.import_module("app_dashboard")
    return module.resolve_api_url()


def test_resolve_api_url_ignores_self_host(monkeypatch):
    result = _call_resolve_api_url(monkeypatch)
    assert result == "http://127.0.0.1:8000"
