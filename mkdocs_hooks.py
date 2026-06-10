"""MkDocs build hooks for the recension docs site.

The example pages under ``docs/examples/generated/`` are produced from real
(offline, deterministic) runs of the example scripts and are intentionally not
committed. This ``on_pre_build`` hook regenerates them before every build, so a
bare ``mkdocs build`` works on a fresh clone with no separate step — the nav can
never reference a missing page.

Registered via ``hooks:`` in ``mkdocs.yml``. Standalone regeneration is still
available with ``make examples`` / ``python examples/build_site.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_EXAMPLES_DIR = Path(__file__).parent / "examples"


def on_pre_build(config: Any) -> None:
    """Regenerate the rendered example pages before the site is built."""
    if str(_EXAMPLES_DIR) not in sys.path:
        sys.path.insert(0, str(_EXAMPLES_DIR))
    import build_site

    build_site.main()
