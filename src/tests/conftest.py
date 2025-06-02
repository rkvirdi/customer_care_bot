# tests/conftest.py

import sys
import pathlib

# 1. File __file__ is …/project-root/src/tests/conftest.py
# 2. parent.parent.parent goes:
#      conftest.py → tests/ → src/ → project-root/
PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent.resolve()

# 3. Insert project root so "import src.main" works
sys.path.insert(0, str(PROJECT_ROOT))
