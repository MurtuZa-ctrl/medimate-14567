"""Embeds model.json into index.src.html -> index.html (single file, no server needed)."""
import json, pathlib
here = pathlib.Path(__file__).parent
model = json.loads((here / "model.json").read_text())
src = (here / "index.src.html").read_text()
assert "/*MODEL_JSON*/null" in src
out = src.replace("/*MODEL_JSON*/null", json.dumps(model))
(here / "index.html").write_text(out)
print("built index.html", len(out), "bytes")
