"""Streamlit Cloud entry point.

Streamlit reruns this file after every widget interaction. Importing app.py with
``from app import *`` only executes the module once because Python caches imports,
which leaves later reruns blank or stuck on the first view. ``run_path`` executes
the app script on every rerun while keeping app.py as the shared implementation.
"""

from pathlib import Path
import runpy


runpy.run_path(str(Path(__file__).with_name("app.py")), run_name="__main__")
