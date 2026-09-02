#!/usr/bin/env bash
# Where the overnight zero run got to.
RUN="${1:-zero1}"
D="$HOME/.local/share/minihouse-zero/zero/$RUN"
echo "== $RUN =="
[ -f "$D/status.json" ] && cat "$D/status.json"
echo "-- per-iteration --"
[ -f "$D/log.jsonl" ] && ./venv/bin/python - "$D/log.jsonl" <<'PY'
import json,sys
for line in open(sys.argv[1]):
    r=json.loads(line); t=r.get("train") or {}; e=r.get("eval") or {}; sp=r.get("selfplay") or {}
    print("iter %2d  %5.1f min  games %-5s  policy %.3f  value %.3f  entropy %.3f  mae %.3f  %s"
          % (r["iteration"], r["seconds"]/60, sp.get("games"), t.get("policy_loss",0),
             t.get("value_loss",0), t.get("entropy",0), t.get("value_mae",0),
             ("vs d%d: %.3f" % (e["depth"], e["score"])) if e else ""))
PY
echo "-- tail --"; tail -5 "$D/run.log" 2>/dev/null
