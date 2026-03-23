import sys
from pathlib import Path

# Allow `import data`, `import models`, ... as in `python main.py` from `aegis/`
_ROOT = Path(__file__).resolve().parent.parent / "aegis"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
