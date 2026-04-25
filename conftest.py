import sys
from pathlib import Path

# Add parent of ml_eval_framework to sys.path so relative imports resolve
root = Path(__file__).parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
