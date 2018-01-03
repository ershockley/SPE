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

# env stuff
source activate pax_head
export PYTHONPATH=/project/lgrandi/anaconda3/envs/pax_head/lib/python3.4/site-packages:$PYTHONPATH
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
#SBATCH --qos=xenon1t
#SBATCH --partition=xenon1t

export PATH=/project/lgrandi/anaconda3/bin:\$PATH
source activate pax_head

if [[ ! -e $tmp_dir/${noise_name} ]]; then
    source /home/ershockley/setup_rucio.sh
    #source /project/lgrandi/general_scripts/setup_rucio.sh
    mkdir $tmp_dir/${noise_name}
    echo "rucio download $noise_DID --dir $tmp_dir/${noise_name} --no-subdir"
    rucio download $noise_DID --dir $tmp_dir/${noise_name} --no-subdir 
fi
EOF

# loop over LED_runs, submit jobs that download LED run and do runs spe_acceptance code
for run in $LED_runs; do
    padded_run=$(printf "%05d" $run)
    if [[ -e $workdir/data/run_${padded_run}.h5 ]]; then
	continue
    fi
    DID=$(python get_rucio_did.py $run)
    name=$(python get_name.py $run)
    sbatch_script=$PWD/sbatch_scripts/$run.sbatch
    cat <<EOF > $sbatch_script
#!/bin/bash
#SBATCH --job-name=spe_$run
#SBATCH --output=${PWD}/logs/run_${run}.log
#SBATCH --error=${PWD}/logs/run_${run}.log
#SBATCH --account=pi-lgrandi
#SBATCH --qos=xenon1t
#SBATCH --partition=xenon1t
#SBATCH --mem=16GB

export PATH=/project/lgrandi/anaconda3/bin:\$PATH

if [[ ! -e $tmp_dir/$name ]]; then
    tmp_pypath=$PYTHONPATH
    source /home/ershockley/setup_rucio.sh
    #source /project/lgrandi/general_scripts/setup_rucio.sh
    mkdir $tmp_dir/$name
    echo "rucio download $DID --dir $tmp_dir/$name --no-subdir"
    rucio download $DID --dir $tmp_dir/$name --no-subdir
    export PYTHONPATH=$tmp_pypath
fi
source activate pax_head
echo "$workdir/spe_acceptance.py $run $noise_run"
$workdir/spe_acceptance.py $run $noise_run
EOF
    
    printf "sbatch $sbatch_script\n" >> $noise_sbatch
done
printf "sleep 10" >> $noise_sbatch

sbatch $noise_sbatch
