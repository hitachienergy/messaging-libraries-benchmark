#!/bin/bash

cd "$(dirname "$0")"/../../

ERROR_LOG_FOLDER="logs"
RESULTS_FOLDER="results"
ERROR_LOG_FILE="$ERROR_LOG_FOLDER/errors.log"
RESULTS_FILE="$RESULTS_FOLDER/results.csv"

mkdir -p $ERROR_LOG_FOLDER $RESULTS_FOLDER

sudo setcap cap_sys_nice+ep $(which chrt)
make -C bml all
mkdir -p $ERROR_LOG_FOLDER $RESULTS_FOLDER
touch $ERROR_LOG_FILE

HEADER="Library,Protocol,Messages Sent,Payload Length,Pub Delay,Pub Interval,Num Subscribers,Messages Received,Total Time,Throughput,Payload Throughput,Min Latency,Avg Latency,P90 Latency,P99 Latency,Max Latency,Mean Jitter,Median CPU,Median MEM"
if [ ! -f $RESULTS_FILE ]; then
    echo "$HEADER" > $RESULTS_FILE
else
    if [ "$(head -n 1 $RESULTS_FILE)" != "$HEADER" ]; then
        echo "Header in $RESULTS_FILE is incorrect. Please remove the file and run the script again."
        exit 1
    fi
fi