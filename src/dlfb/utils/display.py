import inspect
import re
import textwrap
from itertools import islice
from typing import Callable

from IPython.display import Code
from IPython.display import display as ipython_display

DEV_COMMENT_KEYWORDS = r"(TODO|NOTE|QUESTION)"


def display(
  sources: list[Callable | str],
  sep: str = "\n\n",
  hide: list[Callable | str] = [],
) -> str:
  blocks = {
    k: [
      source if isinstance(source, str) else inspect.getsource(source)
      for source in v
    ]
    for k, v in {"sources": sources, "hide": hide}.items()
  }
  source_block = sep.join(block for block in blocks["sources"])
  for block in blocks["hide"]:
    source_block = source_block.replace(block, "")
  clean_code = drop_dev_comments(source_block)
  ipython_display(Code(clean_code, language="python"))


def drop_dev_comments(source_block: str):
  filtered_lines = []
  for line in source_block.splitlines():
    if re.match(rf"\s*#\s*{DEV_COMMENT_KEYWORDS}\b", line, re.IGNORECASE):
      continue
    line = re.sub(
      rf"(#\s*){DEV_COMMENT_KEYWORDS}\b.*", "", line, flags=re.IGNORECASE
    )
    filtered_lines.append(line.rstrip())
  return "\n".join(filtered_lines)


def print_short_dict(d, max_items=10, width=80):
  shown = list(islice(d.items(), max_items))
  remaining = len(d) - len(shown)
  preview = {k: v for k, v in shown}
  s = str(preview)
  wrapped_lines = textwrap.wrap(s, width=width)
  for line in wrapped_lines:
    print(line)
  if remaining > 0:
    print(f"…(+{remaining} more entries)")
