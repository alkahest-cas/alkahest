"""``python -m alkahest.playground`` — run the Math playground server locally."""

import sys

from .server import main

if __name__ == "__main__":
    sys.exit(main())
