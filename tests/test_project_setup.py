from pathlib import Path


def test_vista_package_imports(monkeypatch):
    src_path = str(Path(__file__).resolve().parents[1] / "src")
    monkeypatch.syspath_prepend(src_path)

    import vista

    assert vista is not None
