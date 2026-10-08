import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
# never touch the real ~/LiveTranslator from tests
os.environ["LT_HOME"] = tempfile.mkdtemp(prefix="lt-test-home-")
