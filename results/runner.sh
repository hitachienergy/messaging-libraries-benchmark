#!/bin/bash
##
# This script is used to run the bruteforcer for the BML Optimizer project.
# It sets up the environment, defines parameters for the benchmarking, and executes the bruteforcer script.
# The results are saved in a CSV file.
# 
# You should run this script from the root directory.
#
# To run in the background, use:
#   nohup ./results/runner.sh > bruteforce.log &
# This will allow the script to continue running even if the terminal is closed.
# To run normally, use:
#   ./results/runner.sh
##

pubs=""
for i in $(seq 0 200 1000); do
  pubs="$pubs $i"
done
echo $pubs

#################################################

lengths=""
for i in $(seq 0 9); do
  lengths="$lengths $((2**i*1000))"
done
echo $lengths

set -x
python3 -m bml_optimizer.scripts.bruteforcer \
  --libraries zeromq nanomsg nng \
  --protocols inproc ipc tcp \
  --pub_intervals $pubs \
  --pub_delays 1000 \
  --subscribers 1 2 4 8 \
  --message_counts 5000 \
  --payload_lengths $lengths \
  --runs 4
set +x