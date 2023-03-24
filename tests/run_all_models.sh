#!/bin/bash
# declare -i count=`jq -r '.mlm | length' model_tests.json`
# declare -i count=$count-1

for i in {0..5}
do
    `jq -r '.ner['$i']' tests/model_tests.json`
done

for i in {0..5}
do
    `jq -r '.mlm['$i']' tests/model_tests.json`
done

for i in {0..5}
do
    `jq -r '.sc['$i']' tests/model_tests.json`
done
