#!/usr/bin/env python

#### takes two arguments ####

# 1: LED run number (not blank)
# 2: blank run number
# 3: LED run path
# 4: blank run path

# TODO: find blank run number automatically

import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from pax import core
import multihist
import sys
import os
import scipy.integrate as integrate
from runDB import get_name
#from myRucio import download_raw_files
import stat
import shutil
import pickle

# loop over LED and noise runs, fill histograms
#rawdata_dir = '/project/lgrandi/xenon1t/spe_acceptance/rawdata'
rawdata_dir = '/scratch/midway2/ershockley/rawdata/SPE'
#where processed data will go

data_dir_base = '/project/lgrandi/xenon1t/spe_acceptance/data'

# load run into pax
def get_run(run):
    mypax = core.Processor(config_names='XENON1T', config_dict={
        'pax': {
            'plugin_group_names': ['input', 'preprocessing'],
            'preprocessing':      ['CheckPulses.SortPulses',
                                   'CheckPulses.ConcatenateAdjacentPulses',],
            'input':              'Zip.ReadZipped',
            'encoder_plugin':     None,
            #'decoder_plugin':     'BSON.DecodeZBSON',
            'input_name':          run
        }
    })
    return mypax

# generator used to loop over events
def get_events(RUN):
    for event in RUN.get_events():
        event = RUN.process_event(event)
        yield event


def loop_over_events(LED_num, noise_num):
    n_channels = 254
    amplitude_bounds = (-100, 1000)

    LED_file = os.path.join(rawdata_dir, get_name(LED_num))
    noise_file = os.path.join(rawdata_dir, get_name(noise_num))

    LED_run = get_run(LED_file)
    noise_run = get_run(noise_file)

    LED_events = LED_run.input_plugin.number_of_events
    noise_events = noise_run.input_plugin.number_of_events

    noise_event_generator = get_events(noise_run)
    LED_event_generator = get_events(LED_run)

    n_loop_events = min(LED_events-1, noise_events-1)

    amplitude_bounds = (-100, 1000)
    n_channels = 254

    LED_window = [125, 175]

    noise_good_events_seen = 0
    LED_good_events_seen = 0

    runs = ['noise', 'LED']

    # get first event to check parameters:

    noise_event_0 = next(noise_event_generator)
    LED_event_0 = next(LED_event_generator)


    noise_samples_per_pulse = len(noise_event_0.pulses[0].raw_data)
    LED_samples_per_pulse = len(LED_event_0.pulses[0].raw_data)

    if noise_samples_per_pulse == LED_samples_per_pulse:
        samples_per_pulse = noise_samples_per_pulse
    else:
        print("noise samples per pulse different than LED samples per pulse. Aborting.")
        return

    led_array = [[], [], []]
    noise_array = [[], [], []]

    
    noise_amplitude = []
    noise_charge = []
    led_amplitude = []
    led_charge = []
    
    for event_i in tqdm(range(n_loop_events - 1)):
        for run in runs:
            if run == 'noise':
                event = next(noise_event_generator)

            else:
                event = next(LED_event_generator)

            if not (len(event.pulses) == n_channels):
                # Ignore weird events where not all channels are present
                # These are probably due to a bug in the event builder
                continue

            if run == 'noise':
                noise_good_events_seen += 1
            else:
                LED_good_events_seen += 1

            channel_list = np.ones(n_channels)
            amplitude_list = np.ones(n_channels)  # (len(show_channels))
            charge_list = np.ones(n_channels)

            for ch, p in enumerate(event.pulses):
                w = p.raw_data
                assert len(w) == samples_per_pulse

                w = np.median(w) - w  # Baseline the waveform by subtracting the median, flip signal

                spe = w[LED_window[0]:LED_window[1]]  # consider LED window only
                spe = np.clip(spe, *amplitude_bounds)

                channel_list[ch] = p.channel

                amplitude_list[ch] = max(spe)

                # for the charge spectrum
                charge = integrate.simps(spe)
                charge_list[ch] = charge

            if run == 'noise':
                # noise_multihist.add(channel_list, amplitude_list, charge_list)
                noise_array[0].extend(channel_list)
                noise_array[1].extend(amplitude_list)
                noise_array[2].extend(charge_list)
                #noise_amplitude.append(amplitude_list)
                #noise_charge.append(charge_list)
            else:
                # LED_multihist.add(channel_list, amplitude_list, charge_list)
                led_array[0].extend(channel_list)
                led_array[1].extend(amplitude_list)
                led_array[2].extend(charge_list)
                #led_amplitude.append(amplitude_list)
                #led_charge.append(charge_list)
        

    print("noise: %d proper events seen in %d events" % (noise_good_events_seen, n_loop_events))
    print("LED: %d proper events seen in %d events" % (LED_good_events_seen, n_loop_events))

    LED_multihist = multihist.Histdd(*tuple(led_array),
                                     axis_names=['channel', 'amplitude', 'charge'],
                                     bins=(np.arange(-1, n_channels + 1),
                                           np.arange(*amplitude_bounds),
                                           np.arange(*amplitude_bounds)))

    noise_multihist = multihist.Histdd(*tuple(noise_array),
                                       axis_names=['channel', 'amplitude', 'charge'],
                                       bins=(np.arange(-1, n_channels + 1),
                                             np.arange(*amplitude_bounds),
                                             np.arange(*amplitude_bounds)))


    #df = pd.DataFrame({'noise_amplitude': noise_amplitude,
    #                   'noise_charge': noise_charge,
    #                   'led_amplitude': led_amplitude,
    #                   'led_charge': led_charge})

    # return led_array, noise_array
    #del noise_array
    #del led_array
    #df.to_hdf('update_run%d.hdf' % LED_num, 'data')
    return LED_multihist, noise_multihist

#makes files group readable, writable, executable
def change_permissions(filename):
    #change group of new file
    shutil.chown(filename, group='pi-lgrandi')
    #change permissions)
    os.system('chmod u+rw %s' %filename)
    os.system('chmod g+rw %s' %filename)

def write_to_file(filename, LED_multihist, noise_multihist):
    x = LED_multihist.bin_centers()[2]
    LED_amplitudes = []
    LED_charges = []
    noise_amplitudes = []
    noise_charges = []
    for ch in tqdm(range(248)):
        LED_amplitudes.append(LED_multihist.slice(ch, ch, 'channel').project('amplitude').histogram)
        LED_charges.append(LED_multihist.slice(ch, ch, 'channel').project('charge').histogram)
        noise_amplitudes.append(noise_multihist.slice(ch, ch, 'channel').project('amplitude').histogram)
        noise_charges.append(noise_multihist.slice(ch, ch, 'channel').project('charge').histogram)

    data = pd.DataFrame()
    data['bins'] = x
    data['LED_amplitude'] = list(np.array(LED_amplitudes).T)
    data['LED_charge'] = list(np.array(LED_charges).T)
    data['noise_amplitude'] = list(np.array(noise_amplitudes).T)
    data['noise_charge'] = list(np.array(noise_charges).T)
    filename = os.path.join(data_dir_base, filename)
    #give the directory and the file the right group permissions
    data.to_hdf(filename, key='data')
    change_permissions(filename)
    print("Data written to %s" % filename)
    
def main(args):
    # set logging default to INFO, setup plotting stuff
    logging.basicConfig(level=logging.INFO)

    # make sure we have right number of args
    if len(args) != 2:
        print("2 arguments required, (1) LED run number, (2) noise run number")
        return

    # get LED and blank run numbers from args
    LED_run_number = int(args[0])
    noise_run_number = int(args[1])

    print("LED run: %d" % (LED_run_number))
    print("noise run: %d" % (noise_run_number))

    # PROCESS THE DATA
    #loop_over_events(LED_run_number, noise_run_number)
    LED_hist, noise_hist = loop_over_events(LED_run_number, noise_run_number)

    # write to file
    filename = 'run_%05d.h5' % LED_run_number
    filename = os.path.join(data_dir_base, filename)
    write_to_file(filename, LED_hist, noise_hist)


if __name__ == "__main__":
    main(sys.argv[1:])





