from pathlib import Path

from ragfold.engines import DEFAULT_ENGINE_NAMES

ROOT = Path(__file__).resolve().parents[1]


def test_readme_contains_engine_matrix_and_every_engine():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "## Engine Comparison" in readme
    assert "## How to Choose" in readme
    assert "## Why Ragfold" in readme
    assert "## Install Extras" in readme
    for engine_name in DEFAULT_ENGINE_NAMES:
        assert engine_name in readme


def test_benchmark_docs_describe_methodology_costs_and_datasets():
    docs = (ROOT / "docs" / "benchmarks.md").read_text(encoding="utf-8")

    assert "## Methodology" in docs
    assert "## Dataset Matrix" in docs
    assert "## Cost per 1K Queries" in docs
    assert "research-based estimate" in docs


def test_contributor_agent_changelog_and_rules_are_present():
    for relative_path in [
        "CONTRIBUTING.md",
        "CHANGELOG.md",
        "AGENTS.md",
        "CLAUDE.md",
        "docs/conventions/golden-rules.md",
    ]:
        content = (ROOT / relative_path).read_text(encoding="utf-8")
        assert "TDD" in content or "Test" in content
        assert "Never push" in content or "Do not push" in content


def test_ci_matrix_is_light_and_cross_platform():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "ubuntu-latest" in ci
    assert "macos-latest" in ci
    assert "windows-latest" in ci
    assert '"3.10"' in ci
    assert '"3.11"' in ci
    assert '"3.12"' in ci
    assert 'pip install -e ".[dev]"' in ci
    assert 'pytest -m "not slow"' in ci
