from pathlib import Path


def test_github_rag_optional_extras_are_declared():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    for extra_name in [
        "lightrag",
        "rag-anything",
        "adaptive-chunking",
        "headroom",
        "cocoindex",
        "agentic-file-search",
        "promptfoo",
    ]:
        assert f"{extra_name} = [" in pyproject
