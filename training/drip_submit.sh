#!/bin/bash
# Feeds the sweep into SLURM two jobs at a time.
#
# The mtech QOS caps this account at 2 jobs in the system at once
# (MaxSubmitJobsPerUser=2), so sbatch-ing everything at once gets all but
# two hard-rejected with QOSMaxSubmitJobPerUserLimit. This polls and
# submits only when a slot is free.
#
# Guarded against double-submission: jobs 2249 and 2250 were both
# train_speckle_looks4_s2, submitted two minutes apart by a single feeder.
# The rewrite of the pending file (tail -n +2 > tmp && mv) had failed with
# "Disk quota exceeded" -- the cluster home quota was full -- so the same
# head line was read again on the next poll. The queue file alone is
# therefore not a reliable record of what has been submitted;
# sweep_submitted.txt is, so it is consulted before every submit and the
# entry is written there BEFORE the queue file is rewritten. A crash
# between the two now costs a skipped script rather than a duplicate GPU
# job.
#
# The same exhausted quota silently truncated checkpoints to 0 bytes in
# jobs 2227, 2243 and 2245 while their training logs looked perfect, so a
# write failure here is worth treating as a symptom, not just noise.
set -u
cd /userhome/mtech/akashc1005/VarunaNet || exit 1

QUEUE_FILE=sweep_pending.txt
DONE_FILE=sweep_submitted.txt
MAX_IN_FLIGHT=2
POLL_SECONDS=120

touch "$DONE_FILE"

while :; do
    remaining=$(grep -cve '^[[:space:]]*$' "$QUEUE_FILE" 2>/dev/null || echo 0)
    if [ "$remaining" -eq 0 ]; then
        echo "$(date '+%F %T')  queue empty; exiting"
        break
    fi

    in_flight=$(squeue -h -u "$USER" | wc -l | tr -d ' ')
    if [ "$in_flight" -lt "$MAX_IN_FLIGHT" ]; then
        script=$(head -n 1 "$QUEUE_FILE")
        if [ -n "$script" ]; then
            if grep -qF "  $script" "$DONE_FILE"; then
                echo "$(date '+%F %T')  already submitted, dropping: $script"
                tail -n +2 "$QUEUE_FILE" > "$QUEUE_FILE.tmp" && mv "$QUEUE_FILE.tmp" "$QUEUE_FILE"
                continue
            fi
            jid=$(sbatch --parsable "training/$script" 2>&1)
            if [[ "$jid" =~ ^[0-9]+$ ]]; then
                echo "$(date '+%F %T')  submitted $jid  $script" >> "$DONE_FILE"
                echo "$(date '+%F %T')  submitted $jid  $script"
                tail -n +2 "$QUEUE_FILE" > "$QUEUE_FILE.tmp" && mv "$QUEUE_FILE.tmp" "$QUEUE_FILE"
            else
                echo "$(date '+%F %T')  deferred $script :: $jid"
            fi
        fi
    else
        echo "$(date '+%F %T')  $in_flight in flight, $remaining pending"
    fi
    sleep "$POLL_SECONDS"
done
