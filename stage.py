import os
import sys
import numpy as np
from get_rse import get_rses
from get_rucio_did import get_did
from get_name import get_name
import subprocess

def get_runs():
    # get run numbers from the runlist files
    runs = []
    for tfile in os.listdir('runlists'):
        if not tfile.endswith('.txt'):
            continue

        with open(os.path.join('runlists', tfile)) as f:
            l = [int(i.rstrip()) for i in f.readlines()]
        runs.extend(l)

    return sorted(runs)

    
def rucio_list_files(run_id):
    did = get_did(run_id)
    out = subprocess.Popen(["rucio", "list-file-replicas", did], stdout=subprocess.PIPE).stdout.read()
    out = str(out).split("\\n")
    files = set([l.split(" ")[3] for l in out if '---' not in l and 'x1t' in l])
    zip_files = np.array(sorted([f for f in files if f.startswith('XENON1T')]))
    return zip_files


def make_rucio_command(run, path, sleep=60):
    rses = get_rses(run)
    if len(rses) < 1:
        print("Problem finding rses for run %d" %run)
    did = get_did(run)
    name = get_name(run)
    path = os.path.join(path, name)
    cmd_template = "rucio download {did} --dir {path} --no-subdir {rse}\n"
    
    if "UC_OSG_USERDISK" in rses:
        rse = "--rse UC_OSG_USERDISK"
    elif "NIKHEF_USERDISK" in rses:
        rse = "--rse NIKHEF_USERDISK"
    elif "SURFSARA_USERDISK" in rses:
        rse = "--rse SURFSARA_USERDISK"
    elif "CCIN2P3_USERDISK" in rses:
        rse = "--rse CCIN2P3_USERDISK"
    else:
        rse = ""

    if os.path.exists(path):
        # do we have all of the files?
        command = ""
        for zipfile in rucio_list_files(run):
            if not os.path.exists(os.path.join(path, zipfile)):
                command += cmd_template.format(did=did.replace('raw', zipfile), path=path, rse=rse)

    else:
        command = cmd_template.format(did=did, path=path, rse=rse)
    # add a sleep statement
    if len(command) > 0:
        command += "sleep {sleep}\n".format(sleep=sleep)
    return command


def write_executables(filepath, nfiles=3):

    runs = get_runs()
    print("Staging %d runs" % len(runs))
    runs_per_file = int(np.ceil(len(runs)/nfiles))

    for i in range(nfiles):
        script = """#!/bin/bash
source ~/setup_rucio.sh
"""
        runslice = runs[runs_per_file*i : runs_per_file*(i+1)]
        thisfile = filepath.replace('.', '_%d.'%i)
        for r in runslice:
            cmd = make_rucio_command(r, "/scratch/midway2/ershockley/rawdata/SPE")
            script = script + cmd

        with open(thisfile, 'w') as f:
            f.write(script)


if __name__ == "__main__":
    write_executables(sys.argv[1])
