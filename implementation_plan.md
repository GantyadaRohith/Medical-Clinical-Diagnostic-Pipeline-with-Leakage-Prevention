# UI Container Styling Refactor

Refactor the Streamlit UI dashboard container layouts in `app_dashboard.py`. The dashboard currently uses raw HTML wrappers (`st.markdown('<div class="glass-card">', unsafe_allow_html=True)` and `st.markdown('</div>', unsafe_allow_html=True)`) across different Streamlit calls. Because Streamlit encapsulates each output in its own React DOM container, elements like columns, charts, and metrics render outside (below) the custom styled glassmorphic box.

We will refactor this to use `st.container(key="...")` and target the containers using modern Streamlit CSS class styling (`.st-key-<key>`).

## Proposed Changes

### [Streamlit Dashboard]

#### [MODIFY] [app_dashboard.py](file:///c:/ibm%20prOJECT/app_dashboard.py)

1. **Update CSS Injection (Lines 50–70)**:
   - Map all custom container keys (`concept_card`, `dataset_overview_card`, `secure_results_card`, etc.) to the `.glass-card` styling rules.
   - Define custom styles for height-constrained cards (`roc_card`, `secure_cm_card`, `leaky_cm_card`) and centered cards (`risk_summary_card`).

2. **Refactor Tab 1 (Data Leakage & Security)**:
   - Replace opening/closing of `glass-card` div with `with st.container(key="concept_card"):`.
   - Merge the leaky-box and secure-box markdown blocks to render cleanly in a single `st.markdown` call.

3. **Refactor Tab 2 (Dataset Statistics)**:
   - Replace opening/closing of `glass-card` div with `with st.container(key="dataset_overview_card"):`.

4. **Refactor Tab 3 (Model Comparison)**:
   - Replace opening/closing of secure results card in `col_m1` with `with st.container(key="secure_results_card"):`.
   - Replace opening/closing of leaky results card in `col_m2` with `with st.container(key="leaky_results_card"):`.
   - Replace generalization gap chart card with `with st.container(key="gap_card"):`.
   - Replace ROC curve card with `with st.container(key="roc_card"):`.
   - Replace secure CM card with `with st.container(key="secure_cm_card"):`.
   - Replace leaky CM card with `with st.container(key="leaky_cm_card"):`.
   - Replace feature importance card with `with st.container(key="importance_card"):`.

5. **Refactor Tab 4 (Patient Risk Calculator)**:
   - Replace diagnostic input panel card with `with st.container(key="input_panel_card"):`.
   - Replace mortality risk summary card with `with st.container(key="risk_summary_card"):`.
   - Replace execution trace card with `with st.container(key="trace_card"):`.

---

## Verification Plan

### Automated Tests
We will verify python syntax by running a compiler check:
```powershell
.\venv_std\Scripts\python.exe -m py_compile app_dashboard.py
```

### Manual Verification
1. Run both backend and frontend servers:
   - FastAPI server: `.\venv_std\Scripts\uvicorn.exe app_api:app --host 127.0.0.1 --port 8000`
   - Streamlit frontend: `.\venv_std\Scripts\streamlit.exe run app_dashboard.py --server.port 8501`
2. Open the page and inspect the alignment.
