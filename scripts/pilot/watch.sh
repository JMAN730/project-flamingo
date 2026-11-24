#!/usr/bin/env bash
# The developer's side of Pilot: waits until a [Pilot] report on the fork is
# new or updated, prints it, and exits 0; exits 2 when none came in time.
# Usage: scripts/pilot/watch.sh [seconds, default 1500]
# PILOT_STATUS=1 also wakes on the status issue (it is edited every 10 min).
set -euo pipefail
repo=${PILOT_REPO:-MfDev0/victory}
seen=${PILOT_SEEN:-${TMPDIR:-/tmp}/pilot-seen.txt}
limit=${1:-1500}
touch "$seen"
filter='.[] | select((.title | startswith("[Pilot] status")) | not)'
if [ "${PILOT_STATUS:-0}" = 1 ]; then filter='.[]'; fi
list() {
  gh issue list -R "$repo" --state all --limit 100 --search '"[Pilot]" in:title' \
    --json number,updatedAt,title --jq "$filter"' | "\(.number) \(.updatedAt) \(.title)"'
}
end=$((SECONDS + limit))
while :; do
  now=$(list)
  changed=$(grep -vxF -f "$seen" <<<"$now" || true)
  if [ -n "$changed" ]; then
    echo "== changed Pilot reports"
    echo "$changed"
    while read -r n _; do
      echo
      echo "================ #$n"
      gh issue view "$n" -R "$repo" --comments | head -c 30000
    done <<<"$changed"
    printf '%s\n' "$now" >"$seen"
    exit 0
  fi
  if [ "$SECONDS" -ge "$end" ]; then
    echo "no new Pilot reports in ${limit}s"
    exit 2
  fi
  sleep 30
done
