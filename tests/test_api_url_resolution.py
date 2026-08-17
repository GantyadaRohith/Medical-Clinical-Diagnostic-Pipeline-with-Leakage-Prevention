import os
import importlib


def test_resolve_api_url_ignores_self_host(monkeypatch):
    # Simulate a deployed Streamlit app trying to call its own hostname
    monkeypatch.setenv("API_URL", "https://medical-clinical-diagnostic-pipeline.onrender.com")
    monkeypatch.setenv("RENDER_EXTERNAL_HOSTNAME", "medical-clinical-diagnostic-pipeline.onrender.com")
    monkeypatch.delenv("BACKEND_API_URL", raising=False)

    # Reload module or just import to call the resolved function
    import app_dashboard
    importlib.reload(app_dashboard)
    result = app_dashboard.resolve_api_url()
    assert result == "http://127.0.0.1:8000"


def test_resolve_api_url_allows_different_render_host(monkeypatch):
    # Simulate a deployed Streamlit app (frontend) calling a different backend app on Render
    monkeypatch.setenv("API_URL", "https://my-backend-pipeline.onrender.com")
    monkeypatch.setenv("RENDER_EXTERNAL_HOSTNAME", "my-frontend-dashboard.onrender.com")
    monkeypatch.delenv("BACKEND_API_URL", raising=False)

    import app_dashboard
    importlib.reload(app_dashboard)
    result = app_dashboard.resolve_api_url()
    assert result == "https://my-backend-pipeline.onrender.com"

