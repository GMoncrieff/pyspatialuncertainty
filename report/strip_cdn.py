"""Remove unused CDN script tags (Jupyter widget dependencies) from the rendered report."""

import re
from pathlib import Path

path = Path(__file__).with_name("report.html")
html = path.read_text()
patterns = [
    r'<script src="https://cdn\.jsdelivr\.net/npm/(?:jquery|requirejs)@[^"]+"[^>]*></script>\s*',
    r"<script type=\"application/javascript\">define\('jquery', \[\],"
    r"function\(\) \{return window\.jQuery;\}\)</script>\s*",
]
for p in patterns:
    html = re.sub(p, "", html)
external = re.findall(r'<(?:script|link)[^>]+(?:src|href)="https?://[^"]+"', html)
if external:
    raise SystemExit(f"report.html still loads external resources: {external}")
path.write_text(html)
print("report.html is self-contained")
