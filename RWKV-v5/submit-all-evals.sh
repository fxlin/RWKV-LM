#!/bin/bash
# https://chatgpt.com/share/671a62a0-72d4-8004-988c-5827771780d9

# Usage
# ./submit-all-evals.sh --dry-run    #dry-run mode
# ./submit-all-evals.sh    #actual submission. will detect running jobs and skip them

OUTDIR=/home/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/

# Define dry-run flag
DRY_RUN=false
if [ "$1" == "--dry-run" ]; then
    DRY_RUN=true
fi

RUNS=(
    "04b-pre-x59-16x"
    "04b-pre-x59"
    "1b5-pre-x59"
    "1b5-tunefull-x58"
    "3b-pre-x52"
    "3b-pre-x59"
    "3b-pre-x59-16x"
    "3b-tunefull-x58"
)

# Get the running job names and remove leading/trailing whitespaces
RUNNING_JOBS=$(squeue --format="%.20j" --me | tail -n +2 | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')

# Create an associative array to store the status of each job
declare -A JOB_STATUS

for R in "${RUNS[@]}"; do 
    JOB_NAME="EV-$R"
    
    # Check if a job with the same name is already running
    if echo "$RUNNING_JOBS" | grep -Fxq "$JOB_NAME"; then
        echo "Job $JOB_NAME is already running, skipping..."
        JOB_STATUS["$R"]="AlreadyRunning"
        continue
    fi

    # Print message about submitting the job
    echo "Submitting evaluation job $JOB_NAME..."
    JOB_STATUS["$R"]="JustSubmitted"

    # If not a dry-run, change directory and submit the job
    if [ "$DRY_RUN" = false ]; then
        cd "$OUTDIR/$R"
        ./submit-eval.sh
    else
        echo "(Dry-run mode: Skipping evaluation submission for $JOB_NAME)"
    fi
done

# Print summary table at the end
echo ""
echo "Summary of Evaluation Jobs:"
printf "%-20s %-10s\n" "Job Name" "Status"
printf "%-20s %-10s\n" "--------" "------"

for R in "${RUNS[@]}"; do
    STATUS=${JOB_STATUS["$R"]}
    if [ -z "$STATUS" ]; then
        STATUS="Not Submitted"
    fi
    printf "%-20s %-10s\n" "$R" "$STATUS"
done
