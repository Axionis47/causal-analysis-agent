"""Coverage validation tests for agent modules and retry logic."""

from __future__ import annotations

import os
from pathlib import Path

import coverage
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _find_coverage_file() -> Path | None:
    env_file = os.environ.get("COVERAGE_FILE")
    candidates = []
    if env_file:
        candidates.append(Path(env_file))
    candidates.extend(
        [
            Path.cwd() / ".coverage",
            ROOT / ".coverage",
            ROOT.parent / ".coverage",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_coverage() -> coverage.Coverage:
    data_file = _find_coverage_file()
    if data_file is None:
        pytest.skip("Coverage data not available; run with --cov")
    cov = coverage.Coverage(data_file=str(data_file))
    try:
        cov.load()
    except coverage.CoverageException as exc:
        pytest.skip(f"Coverage data not available: {exc}")
    return cov


def _coverage_percent(cov: coverage.Coverage, path: Path) -> float:
    try:
        _, statements, _excluded, missing, _ = cov.analysis2(str(path))
    except coverage.CoverageException as exc:
        raise AssertionError(f"Coverage data missing for {path}: {exc}") from exc
    total = len(statements)
    if total == 0:
        return 100.0
    executed = total - len(missing)
    return 100.0 * executed / total


def test_agent_file_coverage_thresholds():
    cov = _load_coverage()
    agent_dir = ROOT / "app" / "agents"
    agent_files = [
        path
        for path in agent_dir.glob("*.py")
        if path.name not in {"__init__.py"}
    ]
    assert agent_files

    failures = []
    for path in agent_files:
        percent = _coverage_percent(cov, path)
        if percent < 80.0:
            failures.append(f"{path.name}: {percent:.1f}%")
    assert not failures, "Agent coverage below 80%: " + ", ".join(failures)


def test_retry_and_circuit_breaker_coverage_thresholds():
    cov = _load_coverage()
    retry_file = ROOT / "app" / "services" / "retry.py"
    circuit_file = ROOT / "app" / "services" / "circuit_breaker.py"

    retry_percent = _coverage_percent(cov, retry_file)
    circuit_percent = _coverage_percent(cov, circuit_file)

    assert retry_percent >= 90.0, f"retry.py coverage below 90%: {retry_percent:.1f}%"
    assert circuit_percent >= 90.0, f"circuit_breaker.py coverage below 90%: {circuit_percent:.1f}%"
