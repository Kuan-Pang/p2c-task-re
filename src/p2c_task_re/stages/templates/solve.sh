#!/usr/bin/env bash
# Written by p2c_task_re.stages.package from tests/gold.json (docs/task_format.md §7). Edit gold.json, not this.
set -euo pipefail
mkdir -p "$${OUTPUT_DIR:-/app/output}"
cat > "$${OUTPUT_DIR:-/app/output}/answer.json" <<'P2C_GOLD'
$gold_envelope
P2C_GOLD
