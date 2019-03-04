import subprocess
from runDB import get_did, get_name, get_collection
import os
import numpy as np
import shlex
import argparse

collection = get_collection()

def download_raw(run_id, detector='tpc', rawdir='/scratch/midway2/ershockley/rawdata',
                 rse=None, dry_run=False):
    did = get_did(run_id, detector)
    name = get_name(run_id, detector)
    path = os.path.join(rawdir, name)
    os.makedirs(path, exist_ok = True)
    if dry_run:
        print("DRY RUN - no downloads occurring")
    rse_str = "--rse %s" % rse * (rse is not None)
    command = "rucio download %s --no-subdir --dir %s %s" % (did, path, rse_str)
    if dry_run:
        print(command)
        return
    subprocess.Popen(shlex.split(command)).communicate()


def get_rses(run_id, detector='tpc'):
    name = get_name(run_id, detector)

    doc = collection.find_one({'name': name, 'detector': detector},
                          {'data': 1})

    for datum in doc['data']:
        if datum['host'] == 'rucio-catalogue':
            return datum['rse']
    return []


def main():
    parser = argparse.ArgumentParser(description="Downloads data from rucio")
    parser.add_argument('number', type=int)

    args = parser.parse_args()

    rses = get_rses(args.number)

    if 'UC_OSG_USERDISK' in rses:
        rse = 'UC_OSG_USERDISK'
    elif 'NIKHEF_USERDISK' in rses:
        rse = 'NIKHEF_USERDISK'
    else:
        rse = ""

    print(rse)

if __name__ == "__main__":
    main()
