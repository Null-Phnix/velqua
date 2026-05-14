"""Allow running CLI as: python -m src.cli"""

import sys

from . import main

sys.exit(main())
