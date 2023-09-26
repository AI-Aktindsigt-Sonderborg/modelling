#!/bin/bash
# declare -i count=`jq -r '.mlm | length' model_tests.json`
# declare -i count=$count-1

for i in {0..5}
do
    `jq -r '.ner['$i']' tests/final_model_runs.json`
done
