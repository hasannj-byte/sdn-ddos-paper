"""Flat shim for POX's ext/ directory.

POX's component loader (pox/boot.py) tries "pox.<name>" before "<name>", and
its fallback heuristic only correctly recovers from that failed first attempt
for flat (non-dotted) component names -- a multi-level dotted path like
"src.sdn.pox_mitigation" breaks the fallback and aborts before ever trying the
plain name. Load this flat module instead ("./pox.py pox_mitigation ..."); it
just re-exports the real implementation, which stays under src/sdn/ with the
rest of the codebase.

Deploy: copy/symlink this file into <pox_repo>/ext/pox_mitigation.py, with
SDN_EXP_PATH pointing at the experiments repo root (the directory containing
`src/`).
"""
import os
import sys

SDN_EXP_PATH = os.environ.get("SDN_EXP_PATH", os.path.expanduser("~/sdn-exp"))
if SDN_EXP_PATH not in sys.path:
    sys.path.insert(0, SDN_EXP_PATH)

from src.sdn.pox_mitigation import launch  # noqa: E402,F401
