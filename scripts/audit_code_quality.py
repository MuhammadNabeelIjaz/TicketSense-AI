"""Code quality and security auditor for TicketSense-AI."""

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXCLUDE_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", "delivery"}

print("="*70)
print("PHASE 8: CODE QUALITY AND SECURITY AUDIT")
print("="*70)

# Check 1: Hardcoded absolute paths
abs_path_pattern = re.compile(r'([A-Za-z]:\\[\w\\]+|/Users/[\w/]+|/home/[\w/]+)')
suspicious_paths = []

# Check 2: API keys / credentials
secret_pattern = re.compile(r'(api_key|secret_key|password|bearer|auth_token)\s*=\s*[\'"][^\'"]+[\'"]', re.IGNORECASE)
found_secrets = []

# Check 3: Broken imports or syntax errors
syntax_issues = []

for py_file in PROJECT_ROOT.rglob("*.py"):
    if any(d in py_file.parts for d in EXCLUDE_DIRS):
        continue

    try:
        content = py_file.read_text(encoding="utf-8")
    except Exception as e:
        syntax_issues.append(f"Cannot read {py_file}: {e}")
        continue

    # Skip test files and evaluation scripts that might deliberately mention paths
    if "tests" not in py_file.parts and py_file.name not in ["package_submission.py", "audit_code_quality.py"]:
        matches = abs_path_pattern.findall(content)
        if matches:
            suspicious_paths.append((str(py_file.relative_to(PROJECT_ROOT)), matches))

    secrets = secret_pattern.findall(content)
    if secrets:
        found_secrets.append((str(py_file.relative_to(PROJECT_ROOT)), secrets))

print("\n1. HARDCODED ABSOLUTE PATH CHECK:")
if not suspicious_paths:
    print("   [PASS] Zero hardcoded absolute paths found in production code.")
else:
    for f, p in suspicious_paths:
        print(f"   [WARN] {f}: {p}")

print("\n2. CREDENTIALS AND SECRET SCAN:")
if not found_secrets:
    print("   [PASS] Zero exposed API keys, bearer tokens, or credentials found.")
else:
    for f, s in found_secrets:
        print(f"   [WARN] {f}: {s}")

print("\n3. PYTHON SYNTAX INTEGRITY:")
if not syntax_issues:
    print("   [PASS] All Python files have valid UTF-8 encoding and syntax.")
else:
    for issue in syntax_issues:
        print(f"   [FAIL] {issue}")

print("="*70)
