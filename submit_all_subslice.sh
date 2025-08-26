#!/usr/bin/env bash
# submit_all_subslice.sh - Run 3c and 4c subslice job submissions sequentially.

set -euo pipefail

echo "--> Starting 3-channel subslice job submissions..."
./submit_all_3c_subslice.sh
echo "--> Finished 3-channel subslice job submissions."

echo "--> Pausing for 60 minutes before starting 4-channel jobs..."
sleep $((60 * 60))

echo "--> Starting 4-channel subslice job submissions..."
./submit_all_4c_subslice.sh
echo "--> Finished 4-channel subslice job submissions."

echo "--> All subslice jobs have been submitted." 