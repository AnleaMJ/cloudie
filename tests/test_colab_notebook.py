"""The Colab notebook is a procedure, not a results sheet."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "colab_train.ipynb"


def test_colab_notebook_has_no_saved_results() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    sources: list[str] = []
    assert notebook["cells"], "notebook has no cells"
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        sources.append(source)
        if cell["cell_type"] == "code":
            assert cell.get("outputs") == []
            assert cell.get("execution_count") is None
    text = "\n".join(sources)
    lowered = text.lower()
    for phrase in (
        "src.nasa_tc",
        "src.baselines",
        "cloudie.train",
        "source_split",
        "storm-wise",
        "torchgeo",
        "cuda_visible_devices",
    ):
        assert phrase in lowered
    assert "not an operational" in lowered or "not a forecast" in lowered
    assert "operationally ready" not in lowered
    assert "operational readiness" not in lowered or "not an operational-readiness" in lowered
    assert "mae_kt" not in lowered
    assert "macro-f1" not in text
    assert "macro_f1" not in text
