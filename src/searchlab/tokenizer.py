import re

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the "
    "to was were will with".split()
)


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    """Lowercase, split into alphanumeric tokens, optionally drop stopwords."""
    tokens = _TOKEN_PATTERN.findall(text.lower())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens
