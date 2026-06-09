"""Insert the 72 historical EvalCase entries into the dataset
module, inside the existing _EVAL_CASES_UNSORTED list (right
before its closing `]`). Re-runnable: bails out if the marker is
already present."""
from pathlib import Path

ds = Path("src/soccer_agent/eval/dataset.py")
extra_path = Path("data/ingested_cases.txt")
MARKER = "fd_bundesliga_10-02-2024_leverkusen_v_bayern"
text = ds.read_text()

if MARKER in text:
    print("already ingested — nothing to do")
    raise SystemExit(0)

extra = extra_path.read_text().rstrip("\n")
# Re-indent every line by 4 spaces and ensure each block ends with
# a trailing comma. The ingest script already uses 4-space indent
# and `EvalCase(...)` blocks; we just need commas.
import re
out_lines: list[str] = []
blocks = re.split(r"(?=^\s*EvalCase\()", extra, flags=re.M)
for blk in blocks:
    if not blk.strip():
        continue
    # blk looks like "    EvalCase(\n ... home_goals=3, away_goals=0,\n    ),"
    # Make sure the inner ", " before the closing ")," stays.
    if not blk.rstrip().endswith(","):
        blk = blk.rstrip() + ",\n"
    out_lines.append(blk)

new_block = "".join(out_lines)

# Insert right before the final `]` of _EVAL_CASES_UNSORTED.
# We look for the last "]\n" or "]\r\n" sequence.
import re
matches = list(re.finditer(r"\]\s*\n", text))
if not matches:
    raise SystemExit("dataset.py has no `]\\n` closer")
last_bracket = matches[-1].start()
# Walk back to the start of the line.
line_start = text.rfind("\n", 0, last_bracket) + 1
new_text = text[:line_start] + new_block + text[line_start:]
ds.write_text(new_text)
print(f"appended {new_block.count('EvalCase(')} cases, new size {len(new_text)}")
