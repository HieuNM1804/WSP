#!/bin/bash
# Appends each unseen evaluation of a run to eval/<run>/REPORT.md until the driver exits. Usage: report_watch.sh <run>
RUN=$1; LOG=runs/$RUN/loop.log; OUT=eval/$RUN/REPORT.md; mkdir -p eval/$RUN
[ -f $OUT ] || printf "# Unseen results: %s\n\n| time | checkpoint | scores |\n|---|---|---|\n" "$RUN" > $OUT
while true; do
  grep -E "eval (pass_[0-9]+|final) \(unseen split" $LOG | while read -r line; do
    key=$(echo "$line" | grep -oE "eval (pass_[0-9]+|final)"); grep -q "| ${key#eval } |" $OUT || echo "| $(echo "$line" | cut -c2-20) | ${key#eval } | $(echo "$line" | grep -oE "\{[^}]*\}") |" >> $OUT
  done
  pgrep -f "harness/driver.py .*--run runs/$RUN" >/dev/null || { grep -q "eval final" $LOG && echo "run finished" >> $OUT; exit 0; }
  sleep 60
done
