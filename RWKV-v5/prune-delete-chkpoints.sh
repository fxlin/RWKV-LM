#!/bin/bash

# delete checkpoints every other N epochs. preserve the most recent K checkpoints

# cd out/04b-pre-x59
# ../../delete-chkpoints.sh --dry-run
# ../../delete-chkpoints.sh 

INTERVAL=35  # minimum interval between checkpoints
K=5  # preserve the most recent checkpoints

DRY_RUN=false

# Check for --dry-run option
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
fi

# List all rwkv-XXX.pth files and extract the numbers, excluding non-numeric filenames
files=($(ls rwkv-*.pth | grep -Eo 'rwkv-[0-9]+\.pth' | sed -E 's/rwkv-([0-9]+)\.pth/\1/'))

# Sort the numbers
sorted_files=($(printf "%s\n" "${files[@]}" | sort -n | head -n -$K))

# Initialize a counter for the number of deletions
delete_count=0

# Iterate through the sorted list and delete files where the difference is greater than 10
prev=${sorted_files[0]}
for ((i=1; i<${#sorted_files[@]}; i++)); do
    curr=${sorted_files[$i]}
    # echo $prev $curr
    if (( curr - prev < $INTERVAL )); then
        if $DRY_RUN; then
            #echo "Would delete: rwkv-${curr}.pth"
            ((delete_count++))
        else
            rm "rwkv-${curr}.pth"
        fi
    else
        echo "NOT delete: rwkv-${curr}.pth"
        prev=$curr
    fi
done

# Print the total number of checkpoints after deletion
if $DRY_RUN; then
    remaining_files=$(( ${#files[@]} - delete_count ))
    echo "Total checkpoints before deletion: ${#files[@]}"
    echo "Total checkpoints after deletion: $remaining_files"
fi