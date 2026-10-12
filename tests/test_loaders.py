import pytest

from searchlab.loaders import load_documents


def test_loads_txt_and_md_files(tmp_path):
    (tmp_path / "a.txt").write_text("alpha text", encoding="utf-8")
    (tmp_path / "b.md").write_text("# beta", encoding="utf-8")

    docs = load_documents(tmp_path)

    assert [d.doc_id for d in docs] == ["a", "b"]
    assert docs[0].text == "alpha text"

def test_skips_empty_files(tmp_path):
    (tmp_path / "real.txt").write_text("content", encoding="utf-8")
    (tmp_path / "empty.txt").write_text("", encoding="utf-8")

    docs = load_documents(tmp_path)

    assert [d.doc_id for d in docs] == ["real"]
    assert docs[0].text == "content"


def test_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "nope")
