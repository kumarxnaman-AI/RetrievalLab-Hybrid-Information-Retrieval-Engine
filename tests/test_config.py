import pytest

from searchlab.config import Config


def test_defaults_are_valid():
    config = Config()
    assert config.chunk_size == 500
    assert config.top_k == 10


def test_invalid_bm25_b_raises():
    with pytest.raises(ValueError):
        Config(bm25_b=5)


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        Config(chunk_size=100, chunk_overlap=100)


def test_invalid_log_level_raises():
    with pytest.raises(ValueError):
        Config(log_level="BAD")


def test_log_level_is_case_insensitive():
    config = Config(log_level="info")
    assert config.log_level == "info"

