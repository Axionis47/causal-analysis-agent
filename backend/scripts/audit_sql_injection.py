#!/usr/bin/env python3
"""SQL Injection Audit Script.

This script scans the codebase for potential SQL injection vulnerabilities.
It looks for:
- Raw SQL queries with string concatenation or f-strings
- Use of text() function outside migrations
- Unsafe execute() calls

Run as part of CI/CD pipeline to ensure SQL injection prevention.

Usage:
    python -m scripts.audit_sql_injection

Exit codes:
    0 - No vulnerabilities found
    1 - Potential vulnerabilities detected
"""

from __future__ import annotations

import ast
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator


@dataclass
class Finding:
    """Represents a potential SQL injection finding."""

    file_path: str
    line_number: int
    severity: str  # "high", "medium", "low"
    description: str
    code_snippet: str


@dataclass
class AuditResult:
    """Results of the SQL injection audit."""

    findings: list[Finding] = field(default_factory=list)
    files_scanned: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def has_high_severity(self) -> bool:
        """Check if any high severity findings exist."""
        return any(f.severity == "high" for f in self.findings)

    @property
    def has_medium_severity(self) -> bool:
        """Check if any medium severity findings exist."""
        return any(f.severity == "medium" for f in self.findings)


# Patterns that indicate potential SQL injection
SQL_KEYWORDS = r"(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|EXECUTE|EXEC)"
SUSPICIOUS_PATTERNS = [
    # f-strings with SQL keywords
    (
        re.compile(rf'f["\'].*{SQL_KEYWORDS}.*\{{', re.IGNORECASE),
        "high",
        "F-string containing SQL keyword - potential SQL injection",
    ),
    # String concatenation with SQL keywords
    (
        re.compile(rf'["\'].*{SQL_KEYWORDS}.*["\'].*\+', re.IGNORECASE),
        "high",
        "String concatenation with SQL keyword - potential SQL injection",
    ),
    # .format() with SQL keywords
    (
        re.compile(rf'["\'].*{SQL_KEYWORDS}.*["\']\.format\(', re.IGNORECASE),
        "high",
        ".format() with SQL keyword - potential SQL injection",
    ),
    # % formatting with SQL keywords
    (
        re.compile(rf'["\'].*{SQL_KEYWORDS}.*%[sd].*["\'].*%', re.IGNORECASE),
        "high",
        "% formatting with SQL keyword - potential SQL injection",
    ),
    # Raw execute with string
    (
        re.compile(r'\.execute\([f"\'][^,]+["\']', re.IGNORECASE),
        "medium",
        "execute() with string literal - verify parameterization",
    ),
    # text() function usage (outside migrations is suspicious)
    (
        re.compile(r'text\([f"\'][^)]+\)', re.IGNORECASE),
        "low",
        "text() function usage - ensure this is in a migration or uses bind parameters",
    ),
]

# Directories to exclude from scanning
EXCLUDED_DIRS = {
    "__pycache__",
    ".git",
    "node_modules",
    "venv",
    ".venv",
    "migrations",
    "alembic/versions",
    ".pytest_cache",
    ".mypy_cache",
}

# File patterns to include
INCLUDE_PATTERNS = ["*.py"]


def get_python_files(root_dir: Path) -> Generator[Path, None, None]:
    """Recursively find all Python files to scan.

    Args:
        root_dir: Root directory to start scanning from.

    Yields:
        Path objects for each Python file found.
    """
    for path in root_dir.rglob("*.py"):
        # Check if any parent directory is in excluded list
        parts = path.parts
        if any(excluded in parts for excluded in EXCLUDED_DIRS):
            continue
        yield path


def scan_file(file_path: Path) -> list[Finding]:
    """Scan a single file for potential SQL injection.

    Args:
        file_path: Path to the Python file to scan.

    Returns:
        List of findings for this file.
    """
    findings = []

    try:
        content = file_path.read_text(encoding="utf-8")
        lines = content.split("\n")

        for line_num, line in enumerate(lines, start=1):
            # Skip comments
            stripped = line.strip()
            if stripped.startswith("#"):
                continue

            # Check against suspicious patterns
            for pattern, severity, description in SUSPICIOUS_PATTERNS:
                if pattern.search(line):
                    # Additional context: check if this looks like a migration file
                    if "alembic" in str(file_path) or "migration" in str(file_path).lower():
                        # Reduce severity for migration files
                        severity = "low" if severity == "high" else severity

                    findings.append(
                        Finding(
                            file_path=str(file_path),
                            line_number=line_num,
                            severity=severity,
                            description=description,
                            code_snippet=line.strip()[:100],
                        )
                    )

    except Exception as e:
        print(f"Warning: Could not scan {file_path}: {e}", file=sys.stderr)

    return findings


def scan_for_unsafe_execute(file_path: Path) -> list[Finding]:
    """Use AST to find unsafe execute() calls with string concatenation.

    Args:
        file_path: Path to the Python file to scan.

    Returns:
        List of findings for unsafe execute calls.
    """
    findings = []

    try:
        content = file_path.read_text(encoding="utf-8")
        tree = ast.parse(content)

        for node in ast.walk(tree):
            # Look for method calls
            if isinstance(node, ast.Call):
                # Check if it's an execute call
                if isinstance(node.func, ast.Attribute) and node.func.attr in (
                    "execute",
                    "executemany",
                    "raw",
                ):
                    # Check if the first argument is a BinOp (string concatenation)
                    # or a JoinedStr (f-string)
                    if node.args:
                        first_arg = node.args[0]
                        if isinstance(first_arg, ast.BinOp):
                            findings.append(
                                Finding(
                                    file_path=str(file_path),
                                    line_number=node.lineno,
                                    severity="high",
                                    description="execute() with string concatenation - SQL injection risk",
                                    code_snippet=ast.unparse(node)[:100],
                                )
                            )
                        elif isinstance(first_arg, ast.JoinedStr):
                            findings.append(
                                Finding(
                                    file_path=str(file_path),
                                    line_number=node.lineno,
                                    severity="high",
                                    description="execute() with f-string - SQL injection risk",
                                    code_snippet=ast.unparse(node)[:100],
                                )
                            )

    except SyntaxError:
        pass  # File has syntax errors, skip AST analysis
    except Exception as e:
        print(f"Warning: AST analysis failed for {file_path}: {e}", file=sys.stderr)

    return findings


def run_audit(root_dir: Path) -> AuditResult:
    """Run the SQL injection audit on the codebase.

    Args:
        root_dir: Root directory to scan.

    Returns:
        AuditResult containing all findings.
    """
    result = AuditResult()

    for file_path in get_python_files(root_dir):
        result.files_scanned += 1

        # Pattern-based scanning
        findings = scan_file(file_path)
        result.findings.extend(findings)

        # AST-based scanning for deeper analysis
        ast_findings = scan_for_unsafe_execute(file_path)
        result.findings.extend(ast_findings)

    # Deduplicate findings (same file/line might be caught by multiple patterns)
    seen = set()
    unique_findings = []
    for finding in result.findings:
        key = (finding.file_path, finding.line_number, finding.severity)
        if key not in seen:
            seen.add(key)
            unique_findings.append(finding)

    result.findings = unique_findings

    return result


def print_report(result: AuditResult) -> None:
    """Print the audit report to stdout.

    Args:
        result: The audit result to report.
    """
    print("=" * 80)
    print("SQL INJECTION AUDIT REPORT")
    print("=" * 80)
    print(f"\nFiles scanned: {result.files_scanned}")
    print(f"Total findings: {len(result.findings)}")

    if not result.findings:
        print("\nNo potential SQL injection vulnerabilities found.")
        print("=" * 80)
        return

    # Group by severity
    high = [f for f in result.findings if f.severity == "high"]
    medium = [f for f in result.findings if f.severity == "medium"]
    low = [f for f in result.findings if f.severity == "low"]

    print(f"\n  HIGH: {len(high)}")
    print(f"  MEDIUM: {len(medium)}")
    print(f"  LOW: {len(low)}")

    if high:
        print("\n" + "-" * 40)
        print("HIGH SEVERITY FINDINGS")
        print("-" * 40)
        for finding in high:
            print(f"\n  File: {finding.file_path}:{finding.line_number}")
            print(f"  Issue: {finding.description}")
            print(f"  Code: {finding.code_snippet}")

    if medium:
        print("\n" + "-" * 40)
        print("MEDIUM SEVERITY FINDINGS")
        print("-" * 40)
        for finding in medium:
            print(f"\n  File: {finding.file_path}:{finding.line_number}")
            print(f"  Issue: {finding.description}")
            print(f"  Code: {finding.code_snippet}")

    if low:
        print("\n" + "-" * 40)
        print("LOW SEVERITY FINDINGS (informational)")
        print("-" * 40)
        for finding in low:
            print(f"\n  File: {finding.file_path}:{finding.line_number}")
            print(f"  Issue: {finding.description}")

    print("\n" + "=" * 80)


def main() -> int:
    """Main entry point for the audit script.

    Returns:
        Exit code (0 for success, 1 for findings).
    """
    # Determine the root directory (backend/app)
    script_dir = Path(__file__).parent
    backend_dir = script_dir.parent
    app_dir = backend_dir / "app"

    if not app_dir.exists():
        print(f"Error: Application directory not found: {app_dir}", file=sys.stderr)
        return 1

    print(f"Scanning: {app_dir}")

    result = run_audit(app_dir)
    print_report(result)

    # Return non-zero if high or medium severity findings
    if result.has_high_severity:
        print("\nAUDIT FAILED: High severity findings detected!")
        return 1
    elif result.has_medium_severity:
        print("\nAUDIT WARNING: Medium severity findings detected. Review recommended.")
        return 0  # Warning only, don't fail CI

    print("\nAUDIT PASSED: No significant SQL injection risks detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
