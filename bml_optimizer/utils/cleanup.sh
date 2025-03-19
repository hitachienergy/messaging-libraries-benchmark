#!/bin/bash
cd "$(dirname "$0")"/../../

TMP_USAGES_DIR="./tmp"
TMP_USAGES="$TMP_USAGES_DIR/output_usages_bml.csv"

TMP_OUTPUT="output.log"

make -C bml all clean
rm -f $TMP_OUTPUT $TMP_OUTPUT-* $TMP_USAGES data/* 127.0.0.1* benchmark* > /dev/null

ps -aux | grep 'pub-sub' | awk '{print $2}' | xargs kill -9 > /dev/null 2>&1