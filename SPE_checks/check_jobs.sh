#!/bin/bash

for runlist  in $(ls runlists)
do
    #get run numbers from runlists
    numbers=$(tail -n 3 runlists/$runlist)
    #echo $numbers

    for num in $numbers
    do
	if [ ! -e data/run_$num/histograms.csv ]
	then
	    echo $num
	fi
    done
done
    #add run numbers to an array to loop over
#    for run in "$runs"
#	do	
#	    numlist+="$run"
#    done
    
    #get run numbers from data
#    cd ./data
#    dataruns=grep  --include="run*"
#    for datarun in "$dataruns"
#    do
#	datanum=grep $dataruns |cut -c "run*" -f2
#    done
    
    #compare numbers from runlist and data
#    for number in "$numbers"
#    do#
#	if [ -e "$number"  in "$datanum" ]
#	then
#	   continue
#	else
#	   echo "$number not found"
#	fi
 #   done  

    #i dont think i need the rest but didnt want to delete it 
	    
#   for runnum in $runlist
#   do
#	if [-f "$./data/*/histogram.csv" in ]; then
    #	    continue
        #print what is missing	
#	else
    #	    echo "$runlist not found"
    #   fi
#   done
#done
