#!/bin/bash

for txt in $(ls runlists/runlist_9*txt)
do
    ./submit_jobs.sh $txt
    sleep 600
done
