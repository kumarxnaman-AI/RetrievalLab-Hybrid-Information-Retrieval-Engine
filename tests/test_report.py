from searchlab.report import save_and_print


def test_report_is_saved_even_when_the_console_cannot_print_it(tmp_path, monkeypatch, capsys):
    real_print = print
    calls = []

    def picky_print(text="", *a, **k):  # behaves like a cp1252 console on the first, non-ASCII print
        calls.append(text)
        if len(calls) == 1:
            raise UnicodeEncodeError("charmap", text, 0, 1, "cannot encode")
        real_print(text, *a, **k)

    monkeypatch.setattr("builtins.print", picky_print)
    out = tmp_path / "sub" / "report.md"
    save_and_print("Delta change: Δ nDCG", out)

    assert out.read_text(encoding="utf-8") == "Delta change: Δ nDCG"  # saved intact
    assert len(calls) == 2 and "?" in calls[1]  # fell back to a printable version
