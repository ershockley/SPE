#!/bin/bash

for txt in $(ls ./runlists/runlist_*txt)
do
    ./submit_jobs.sh $txt
    sleep 300
done
