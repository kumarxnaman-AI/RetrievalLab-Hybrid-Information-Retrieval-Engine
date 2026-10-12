from searchlab.tokenizer import tokenize


def test_lowercases_and_splits_on_punctuation():
    assert tokenize("Pension-Plan, Eligibility!") == ["pension", "plan", "eligibility"]


def test_removes_stopwords_by_default():
    assert tokenize("the plan of the employee") == ["plan", "employee"]


def test_can_keep_stopwords():
    assert tokenize("the plan", remove_stopwords=False) == ["the", "plan"]


def test_empty_text_gives_no_tokens():
    assert tokenize("") == []
