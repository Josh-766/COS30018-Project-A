"""Lets you run `python -m observation` to see the metrics from the logs."""
import sys

from .observation import main

sys.exit(main())
