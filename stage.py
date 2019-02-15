import os
import sys
import numpy as np
from get_rse import get_rses
from get_rucio_did import get_did
from get_name import get_name

def get_runs():
    # get run numbers from the runlist files
    runs = []
    for tfile in os.listdir('runlists'):
        if not tfile.endswith('.txt'):
            continue

        l = tfile.split('.txt')[0].split('_')[1:]
        l = [int(i) for i in l]
        runs.extend(l)

    return sorted(runs)


def make_rucio_command(run, path):
    rses = get_rses(run)
    if len(rses) < 1:
        print("Problem finding rses for run %d" %run)
    did = get_did(run)
    name = get_name(run)
    path = os.path.join(path, name)

    if "UC_OSG_USERDISK" in rses:
        rse = "--rse UC_OSG_USERDISK"
    elif "NIKHEF_USERDISK" in rses:
        rse = "--rse NIKHEF_USERDISK"
    else:
        rse = ""
    command = "rucio download {did} --dir {path} --no-subdir {rse}".format(did=did, path=path, rse=rse)
    return command


def write_executables(filepath, nfiles=5):

    runs = get_runs()
    runs_per_file = int(np.ceil(len(runs)/nfiles))

    for i in range(nfiles):
        script = """#!/bin/bash
source ~/setup_rucio.sh
"""
        runslice = runs[runs_per_file*i : runs_per_file*(i+1)]
        thisfile = filepath.replace('.', '_%d.'%i)
        for r in runslice:
            cmd = make_rucio_command(r, "/scratch/midway2/ershockley/rawdata/SPE")
            script = script + cmd + "\n"

        with open(thisfile, 'w') as f:
            f.write(script)


if __name__ == "__main__":
    write_executables(sys.argv[1])