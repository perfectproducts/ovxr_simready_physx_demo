"""VR SimReady, Newton edition: ovxr + ovrtx + ovstage + ovnewton in one Python process.

The content modules (catalog, voice, browser, HUD text, toaster FMUs, environment generation) are
shared with the C++ app's sidecar and imported from the repository's ``vrsim`` package.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
