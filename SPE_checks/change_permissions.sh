#!/bin/bash

#changes the group of existing data and data folder  to pi-lgrandi
#makes the files group read and writable

cd /project/lgrandi/xenon1t/spe_acceptance
chown :pi-lgrandi data
chmod g+w data

cd data
for file in $(ls data)
do
    chown :pi-lgrandi $file
    chmod g+w $file
done
	    
