"""Reader for the pinned WikiText-2 raw parquet splits."""

from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from tinymem.data.schema import SUPPORTED_SPLITS


@dataclass(frozen=True)
class WikiTextSplit:
    split: str
    rows: tuple[str, ...]
    source_path: Path

    @property
    def text(self) -> str:
        """Return the official rows as one newline-delimited language stream."""
        return "\n".join(self.rows)


def load_wikitext_parquet(path: str | Path, *, split: str) -> WikiTextSplit:
    """Load one official split and preserve every source row in order."""
    if split not in SUPPORTED_SPLITS:
        raise ValueError(f"unsupported split {split!r}")
    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(f"WikiText source file does not exist: {source_path}")
    table = pq.read_table(source_path)
    if table.column_names != ["text"]:
        raise ValueError("WikiText parquet must contain only one 'text' column")
    rows = tuple(table.column("text").to_pylist())
    if not rows or any(not isinstance(row, str) for row in rows):
        raise ValueError("WikiText text column must contain strings")
    return WikiTextSplit(split, rows, source_path)
