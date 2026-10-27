import os
import subprocess
import sys
from pathlib import Path

import yaml

SRC = Path(__file__).resolve().parents[1] / "src"

SCRIPT = """
import sys
from pathlib import Path

for name in ("fast_simplification", "scipy", "skimage", "torch"):
    sys.modules[name] = None

from arena_assets.impl.collider import ColliderError
from arena_assets.impl.collider_gen import generate

try:
    generate(Path(sys.argv[1]))
except ColliderError as error:
    print(error)
"""


def test_generate_without_the_collider_extra_names_the_extra(tmp_path):
    (tmp_path / "annotation.yaml").write_text(yaml.safe_dump({"name": "chair"}))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(SRC), *filter(None, [env.get("PYTHONPATH")])])
    proc = subprocess.run([sys.executable, "-c", SCRIPT, str(tmp_path)], capture_output=True, text=True, env=env, check=False)
    assert proc.returncode == 0, proc.stderr
    assert "[collider] extra" in proc.stdout
