#!/bin/bash

###  takes 1 argument ###
# 1: name (and path if necessary) to runlist txt file

### format of runlist txt file ###
# 1. noise
# 2. bottom run
# 3. top bulk run
# 4. top ring run


# check if log dir exists
if [[ ! -e ./logs ]]; then
    mkdir logs
fi

workdir=$PWD
noise_run=$(head -n 1 $1)
LED_runs=$(tail -n +2 $1)

echo "noise run: $noise_run"
echo "LED runs: $LED_runs"

#where we will save the raw data
tmp_dir="/scratch/midway2/ershockley/rawdata/SPE"
#tmp_dir="/project/lgrandi/xenon1t/spe_acceptance/rawdata"

# env stuff
source activate pax_v6.8.0
#export PYTHONPATH=/project/lgrandi/anaconda3/envs/pax_v6.8.0/bin/python$PYTHONPATH
noise_DID=$( python get_rucio_did.py $noise_run)
noise_name=$( python get_name.py $noise_run)

# write sbatch script for the noise run. This downloads noise run and submits 3 LED jobs.

noise_sbatch=sbatch_scripts/${noise_run}.sbatch
cat <<EOF > $noise_sbatch
#!/bin/bash
#SBATCH --job-name=spe_$noise_run
#SBATCH --output=${PWD}/logs/run_${noise_run}.log
#SBATCH --error=${PWD}/logs/run_${noise_run}.log
#SBATCH --account=pi-lgrandi
#SBATCH --qos=dali
#SBATCH --partition=dali

export PATH=/project/lgrandi/anaconda3/bin:\$PATH
source activate pax_v6.8.0


if [[ ! -e $tmp_dir/${noise_name} ]]; then
    source deactivate
    source /cvmfs/oasis.opensciencegrid.org/osg-software/osg-wn-client/3.3/current/el7-x86_64/setup.sh
    source /cvmfs/xenon.opensciencegrid.org/software/rucio-py27/setup_rucio_1_8_3.sh
    source /project/lgrandi/general_scripts/setup_rucio.sh
    mkdir $tmp_dir/${noise_name}
    chgrp pi-lgrandi $tmp_dir/${noise_name}
    echo "rucio download $noise_DID --dir $tmp_dir/${noise_name} --no-subdir"
    rucio download $noise_DID --dir $tmp_dir/${noise_name} --no-subdir
else
    if [[ -z "\$(ls $tmp_dir/${noise_name})" ]]; then
        source deactivate
        source /cvmfs/xenon.opensciencegrid.org/software/rucio-py27/setup_rucio_1_8_3.sh
        source /cvmfs/oasis.opensciencegrid.org/osg-software/osg-wn-client/3.3/current/el7-x86_64/setup.sh
        source /project/lgrandi/general_scripts/setup_rucio.sh
        mkdir $tmp_dir/${noise_name}
        chgrp pi-lgrandi $tmp_dir/${noise_name}
        echo "rucio download $noise_DID --dir $tmp_dir/${noise_name} --no-subdir"
        rucio download $noise_DID --dir $tmp_dir/${noise_name} --no-subdir
    fi
fi
EOF

# loop over LED_runs, submit jobs that download LED run and do runs spe_acceptance code
for run in $LED_runs; do
    padded_run=$(printf "%05d" $run)
    #need to change this next line to $/project/lgrandi/data/run_${padded_run}.h5?
    if [[ -e /project/lgrandi/xenon1t/spe_acceptance/data/run_${padded_run}.h5 ]]; then
	    echo "data exists"
	    continue
    fi
    DID=$(python get_rucio_did.py $run)
    name=$(python get_name.py $run)
    rse=$(python get_rse.py $run)

    if [[ -n "$rse" ]]; then
        rse="--rse $rse"
    fi
    sbatch_script=$PWD/sbatch_scripts/$run.sbatch
    cat <<EOF > $sbatch_script
#!/bin/bash
#SBATCH --job-name=spe_$run
#SBATCH --output=${PWD}/logs/run_${run}.log
#SBATCH --error=${PWD}/logs/run_${run}.log
#SBATCH --account=pi-lgrandi
#SBATCH --qos=dali
#SBATCH --partition=dali
#SBATCH --mem=24GB

#/home/ershockley/cvmfs_cache_issue.sh
export PATH=/project/lgrandi/anaconda3/bin:\$PATH

if [[ ! -e $tmp_dir/$name ]]; then
    tmp_pypath=$PYTHONPATH
    #source /cvmfs/xenon.opensciencegrid.org/software/rucio-py27/setup_rucio_1_8_3.sh
    source /project/lgrandi/general_scripts/setup_rucio.sh
    mkdir $tmp_dir/$name
    chgrp pi-lgrandi $tmp_dir/$name
    echo "rucio download $DID --dir $tmp_dir/$name --no-subdir"
    rucio download $DID --dir $tmp_dir/$name --no-subdir
    export PYTHONPATH=$tmp_pypath
else
    if [[ -z "\$(ls $tmp_dir/${name})" ]]; then
        #source /cvmfs/xenon.opensciencegrid.org/software/rucio-py27/setup_rucio_1_8_3.sh
        source /project/lgrandi/general_scripts/setup_rucio.sh
        echo "rucio download $DID --dir $tmp_dir/${name} --no-subdir"
        rucio download $DID --dir $tmp_dir/${name} --no-subdir
    fi
fi
source activate pax_v6.8.0 # pax_dev

echo "python $workdir/spe_acceptance.py $run $noise_run"
python $workdir/spe_acceptance.py $run $noise_run
EOF
    
    printf "sbatch $sbatch_script\n" >> $noise_sbatch
done
printf "sleep 10" >> $noise_sbatch

sbatch $noise_sbatch