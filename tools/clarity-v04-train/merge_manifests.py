"""Merge per-video prepare_v04 outputs (root/v*/manifest.jsonl) into root/manifest.jsonl."""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
n = 0
with (root / "manifest.jsonl").open("w") as out:
    for sub in sorted(root.glob("v*/manifest.jsonl")):
        for line in sub.read_text().splitlines():
            r = json.loads(line)
            for k in ("target", "hud", "input"):
                r[k] = f"{sub.parent.name}/{r[k]}"
            out.write(json.dumps(r) + "\n")
            n += 1
print(n, "rows")
