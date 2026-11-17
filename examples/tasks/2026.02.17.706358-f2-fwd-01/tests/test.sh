#!/usr/bin/env bash
# Written by p2c_task_re.stages.package (docs/task_format.md §7): grade the solver's answer with grade.py.
set -euo pipefail
TESTS_DIR="${TESTS_DIR:-/tests}"
mkdir -p "${VERIFIER_DIR:-/logs/verifier}"
python3 "$TESTS_DIR/grade.py" --tests-dir "$TESTS_DIR" \
  --answer "${ANSWER_PATH:-/app/output/answer.json}" --out-dir "${VERIFIER_DIR:-/logs/verifier}"
