"""Vercel Python entrypoint.

The runtime imports ``app`` from this module. Set
``tool.vercel.entrypoint = "app:app"`` in ``pyproject.toml`` as well.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudie.web import create_app

app = create_app()
