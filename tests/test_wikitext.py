from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tinymem.data.wikitext import load_wikitext_parquet


def test_load_wikitext_preserves_rows_and_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "train.parquet"
    pq.write_table(pa.table({"text": ["first", "", "third"]}), path)
    split = load_wikitext_parquet(path, split="train")
    assert split.rows == ("first", "", "third")
    assert split.text == "first\n\nthird"


def test_load_wikitext_rejects_wrong_schema(tmp_path: Path) -> None:
    path = tmp_path / "train.parquet"
    pq.write_table(pa.table({"content": ["text"]}), path)
    with pytest.raises(ValueError, match="one 'text' column"):
        load_wikitext_parquet(path, split="train")
