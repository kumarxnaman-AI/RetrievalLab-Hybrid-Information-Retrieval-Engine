from pathlib import Path


def save_and_print(report: str, path: Path | str) -> None:
    """Write the report as UTF-8 first, then print it.

    Saving comes first so a console that can't display a character (Windows consoles often use
    cp1252) can never cost us a finished experiment. Printing falls back to replacing such characters.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    try:
        print("\n" + report)
    except UnicodeEncodeError:
        print("\n" + report.encode("ascii", "replace").decode("ascii"))
