"""Thin re-export. Canonical window code is ``src.sequences``."""

from src import sequences as _canonical

for _name in dir(_canonical):
    if _name.startswith("__"):
        continue
    globals()[_name] = getattr(_canonical, _name)

del _canonical
del _name
