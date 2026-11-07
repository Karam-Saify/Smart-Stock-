from pathlib import Path


def test_dashboard_files_and_documented_entrypoint_exist():
    root = Path(__file__).resolve().parents[1]
    assert (root / "app.py").exists()
    assert (root / "requirements-ui.txt").exists()
    assert "streamlit run app.py" in (root / "docs" / "DASHBOARD.md").read_text()
