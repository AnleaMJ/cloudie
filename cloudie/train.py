"""Thin re-export. Canonical training code is ``src.train``."""

from src import train as _canonical

for _name in dir(_canonical):
    if _name.startswith("__"):
        continue
    globals()[_name] = getattr(_canonical, _name)

del _canonical
del _name

if __name__ == "__main__":
    main()
