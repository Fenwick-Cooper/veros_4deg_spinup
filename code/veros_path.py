"""
Find the Veros source code and put it on the Python path.

The path is read from veros_path.txt in the repository root: one line, the directory that
contains the `veros` package (i.e. <path>/veros/__init__.py exists). If the file is missing,
empty or wrong, the user is asked for the path, which is checked and then saved there.
Call ensure_veros() before anything imports veros.
"""

import os
import sys

from paths import REPO_DIR

VEROS_PATH_FILE = os.path.join(REPO_DIR, "veros_path.txt")


def _is_veros_source(path):
    return os.path.isfile(os.path.join(path, "veros", "__init__.py"))


def _read():
    if not os.path.isfile(VEROS_PATH_FILE):
        return None
    with open(VEROS_PATH_FILE) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                return os.path.abspath(os.path.expanduser(line))
    return None


def ensure_veros():
    """Return the Veros source directory, asking the user for it if necessary, and put it
    first on sys.path so that this copy of Veros is the one imported."""
    path = _read()
    if path is None or not _is_veros_source(path):
        if path is not None:
            print(f"veros_path.txt gives {path}, but {os.path.join(path, 'veros', '__init__.py')} does not exist.")
        else:
            print(f"The location of the Veros source code is not set ({VEROS_PATH_FILE}).")
        if not sys.stdin.isatty():
            sys.exit("Write the directory that contains the Veros package (the one holding veros/__init__.py) into veros_path.txt and run again.")
        while True:
            answer = input("Directory of the Veros source code (it contains the folder 'veros'): ").strip()
            path = os.path.abspath(os.path.expanduser(answer))
            if _is_veros_source(path):
                break
            print(f"  {os.path.join(path, 'veros', '__init__.py')} not found; try again.")
        with open(VEROS_PATH_FILE, "w") as f:
            f.write(path + "\n")
        print(f"saved to {VEROS_PATH_FILE}")
    if path not in sys.path:
        sys.path.insert(0, path)
    return path
