import matplotlib
matplotlib.use('Agg')
matplotlib.rc('font', size=16)
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import patches as mpatches
import re
import math
from tqdm import tqdm
import numpy as np
from pax.configuration import load_configuration
import hax
from hax.pmt_plot import plot_on_pmt_arrays, pmt_data
from channel_dict import channel_dict
from DB_stuff import find_regular_run
import sys
import os

# see here  https://xecluster.lngs.infn.it/dokuwiki/doku.php?id=xenon:xenon1t:analysis:firstresults:data_quality_pmt
# not including low acceptance PMTs
excluded_pmts = [1, 12, 26, 34, 62, 65, 79, 86, 88, 102, 118, 130, 134, 135, 148,
                 150, 152, 162, 178, 183, 198, 206, 213, 214, 234, 239, 244, 27, 91, 167, 203]

data_dir_base = "./data/"

bottom_runs = []
bottom_occs = []

topb_runs = []
topb_occs = []

topring_runs = []
topring_occs = []


def get_channels(run_number):
    # gets list of channel numbers from csv file
    data_dir = data_dir_base + "run_" + str(run_number)
    file_str = data_dir + "/thresholds.csv"
    channels = []
    with open(file_str) as f:
        for i, l in enumerate(f):
            if i == 0:
                pass
            else:
                line = l.rstrip().split(',')
                ch = int(line[0])
                channels.append(ch)
    return channels


def get_data_array(run_number, hist_str, errors=False):
    # gets specified data array from csv file, with optional errors flag
    data_dir = data_dir_base + "run_" + str(run_number)
    file_str = data_dir + "/histograms.csv"
    channels = get_channels(run_number)
    hist = np.array([])
    amplitudes = []
    errs = np.array([])

    with open(file_str) as f:
        first_line = f.readline().rstrip().split(',')
        amp_index = first_line.index('amplitude')
        hist_index = first_line.index(hist_str)
        if errors:
            err_index = first_line.index(hist_str + 'err')

        newline = f.readline().rstrip().split(',')

        for ch in channels:
            ydata = []
            errdata = []
            line_ch = int(newline[0])
            # print(line_ch)

            while line_ch == ch:
                ydata.append(float(newline[hist_index]))

                if errors:
                    errdata.append(float(newline[err_index]))

                if ch == channels[0]:
                    amplitudes.append(float(newline[amp_index]))

                newline = f.readline().rstrip().split(',')

                if newline == ['']:
                    break

                line_ch = int(newline[0])

            if ch == channels[0]:
                hist = np.array([ydata])
                if errors:
                    errs = np.array([errdata])
            else:
                ydata = np.array([ydata])
                try:
                    hist = np.concatenate((hist, ydata), axis=0)
                except:
                    # print(ch)
                    # print("hist:", len(hist[0]))
                    # print("ydata:", len(ydata[0]))
                    # print(ydata)
                    raise
                if errors:
                    errdata = np.array([errdata])
                    errs = np.concatenate((errs, errdata), axis=0)
    if errors:
        return amplitudes, hist, errs
    else:
        return amplitudes, hist


def get_corrections(run_number):
    # gets corrections from each channel, used to calculate occupancy
    data_dir = data_dir_base + "run_" + str(run_number)
    file_str = data_dir + "/histograms.csv"
    channels = []
    corrections = []
    ch = -99
    with open(file_str) as f:
        first_line = f.readline().rstrip().split(',')

        corr_index = first_line.index('NOISEcorr')

        for line in f:
            line = line.rstrip().split(',')
            if line[0] != ch:
                ch = line[0]
                channels.append(ch)
                corrections.append(float(line[corr_index]))

    return channels, corrections

def get_occ(corrections):
    corr=[]
    occ_list=[]
    occ_array=np.array()
    for i in corrections:
        real_corr=i[1]
        for elem in real_corr:
            corr.append(elem)
    for i in corr:
        occ_array=-1 * math.log(i)
    return occ_array

def plot_occ(run_number, occ_list):
    #plots occupancies as a function of run number

    #divide runs into three types
    bottom_run=[]
    topbulk_run=[]
    topring_run=[]

    for file, runlist in enumerate(sorted(runlist_files)):
        runlist = runlist.split('/')[-1].split('_')
        bottom_run.append(int(runlist[1]))
        topbulk_run.append(int(runlist[2]))
        topring_run.append(int(runlist[3][:-4]))

    #data for plotting
    xaxis=run_number
    yaxis=occ_list


    fig, ax = plt.subplots()

    #base color and label on what kind of run it is
    for run in run_number:
        if run in bottom_run:
            col='r'
            lab='Bottom Run'
        elif run in topbulk_run:
            col='g'
            lab="Topbulk Run"
        elif run in topring_run:
            col='b'
            lab="Topring Run"


    #make legend
    #botpatch=mpatches.Patch(color='red', label="Bottom Run")
    #tbpatch=mpatches.Patch(color='green', label="Topbulk Run")
    #trpatch=mpatches.Patch(color='blue', label="Topring Run")

    #plt.legend(handles=[botpatch, tbpatch, trpatch])

    #set labels
    ax.set(xlabel='Run Number', ylabel='Occupancy',
           title='Occupancy as a Function of Run Number')

    # make scatter plot
    plt.scatter(xaxis, yaxis, color=col, label=lab)

    fig.savefig("occupancy%d.png" % (run_number) )
    plt.show()


def plot_channel(ch, run_number, xlims, ylims = (-100, 500), filedir = ''):
    # plots LED, noise, residual spectrum, acceptance as function of amplitude
    data_dir = data_dir_base + "run_" + str(run_number)
    file_str = data_dir + "/histograms.csv"
    amplitudes, LED_window, LED_err = get_data_array(run_number, "LED", errors=True)
    amplitudes, noise_window, noise_err = get_data_array(run_number, "NOISE", errors=True)
    amplitudes, residual, res_err = get_data_array(run_number, "residual", errors=True)

    plt.figure(figsize=(10,8))
    plt.errorbar(amplitudes, LED_window[ch], yerr=LED_err[0], color='red', linestyle='none',
                 marker='.', label='LED window')
    plt.errorbar(amplitudes, noise_window[ch], yerr=noise_err[0], color='black', linestyle='none',
                 marker='.', label='noise window after correction')
    plt.yscale('log')
    plt.xlabel('amplitude [ADC counts]')
    plt.ylabel('counts')
    plt.legend(loc='upper right', frameon=False)
    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.grid(b=True, which='both')
    plt.title("Channel %d" % ch)
    if filedir != '':
        plt.savefig("%s/ch%d_LEDnoise.png" % (filedir, ch))

    fig, ax1 = plt.subplots(figsize=(10,8))
    ax1.errorbar(amplitudes, residual[ch], yerr=res_err[0], color='blue', linestyle='none')
    ax1.set_yscale('linear')
    ax1.set_xlabel('amplitude [ADC counts]')
    ax1.set_ylabel('LED - noise residual [counts]')
    ax1.set_xlim(xlims)
    plt.title('LED - noise window residual channel %d'% ch)
    plt.grid(b=True, which='both')
    if filedir != '':
        plt.savefig("%s/ch%d_LEDnoiseresidual.png" % (filedir, ch))

    ax2b = ax1.twinx()
    ax2b.set_xlim(xlims)
    ax2b.plot(amplitudes, 1 - (np.cumsum(residual[ch]) / residual[ch].sum()),
              color='red', linewidth=2, linestyle='steps-post')
    ax2b.set_ylabel('Acceptance fraction')
    ax2b.yaxis.label.set_color('red')
    ax1.yaxis.label.set_color('blue')
    for tl in ax1.get_yticklabels():
        tl.set_color('b')
    for tl in ax2b.get_yticklabels():
        tl.set_color('r')
    if filedir != '':
        plt.savefig("%s/ch%d_LEDnoiseresidualACC.png" % (filedir, ch))
    plt.show()

def get_thresholds(run_number):
    # thanks Jelle
    pax_config = load_configuration('XENON1T')
    hax.init()
    run_doc = hax.runs.get_run_info(run_number)

    lookup_pmt = {(x['digitizer']['module'], x['digitizer']['channel']): x['pmt_position']
                  for x in pax_config['DEFAULT']['pmts']}

    baseline = 16000  # This must be in run doc somewhere too... but I guess we didn't change this (much)

    def register_value(r):
        return baseline - int(r['value'], 16)

    thresholds = {}

    for r in run_doc['reader']['ini']['registers']:

        if r['register'] == '8060':
            default_threshold = register_value(r)

        m = re.match(r'1(\d)60', r['register'])
        if m:
            board = int(r['board'])
            channel = int(m.groups()[0])
            threshold = register_value(r)
            try:
                pmt = lookup_pmt[board, channel]
            except KeyError:
                continue
            thresholds[pmt] = threshold

    return [thresholds.get(i, default_threshold)
       for i in np.arange(len(lookup_pmt))]

def twoplus_contribution(occ):
    return 1 - np.exp(-occ)*(1+occ)

def get_acceptances(run_number, thresholds):
    # takes a run number and list of 248 threshold values, returns acceptance value for each channel

    channels = get_channels(run_number)
    amplitudes, acceptance = get_data_array(run_number, 'acceptance')
    # find bin where amplitude is 0
    bin0 = [b for b, x in enumerate(amplitudes) if x == 0.5][0]

    acceptance_fraction = np.ones(len(channels))
    # bad notation, acceptance_fraction is number for a given threshold, and
    # acceptance above is function of amplitude
    for i, ch in enumerate(channels):
        if ch in excluded_pmts:
            acceptance_fraction[i] *= 0
        else:
            acceptance_fraction[i] *= acceptance[i][bin0 + thresholds[i]]

    return acceptance_fraction


def get_acceptances_3runs(bottom_run, top_outer_run, top_bulk_run, thresholds, plot = False):
    # need this for using 3 different runs for acceptance calculation

    runs = [bottom_run, top_outer_run, top_bulk_run]
    # check if channel list are the same for all three runs
    if not (get_channels(bottom_run) == get_channels(top_outer_run) == get_channels(top_bulk_run)):
        print("Channel lists not equal, aborting.")
        return

    # if channel lists identical, just pick one of them to use
    channels = get_channels(bottom_run)

    # define now, will initialize below
    actual_acceptance = None
    actual_amplitude = None  # use 'actual' for consistency, really just the same as amplitudes

    # splice together acceptances from 3 runs into one array
    for run in runs:
        amplitudes, acceptance = get_data_array(run, 'acceptance')

        # if first run in list, initialize the 'actuals'
        if run == bottom_run:
            actual_acceptance = np.zeros(acceptance.shape)
            actual_amplitude = amplitudes.copy()

        for i, ch in enumerate(channels):
            if ch in excluded_pmts:
                continue
            if run == bottom_run and ch in channel_dict["bottom_channels"]:
                actual_acceptance[i] = acceptance[i]

            elif run == top_outer_run and ch in channel_dict["top_outer_ring"]:
                actual_acceptance[i] = acceptance[i]

            elif run == top_bulk_run and ch in channel_dict["top_bulk"]:
                actual_acceptance[i] = acceptance[i]

    # plot functional form of acceptances, with different colors for different runs
    data_dir = data_dir_base
    for ch in channels:
        if ch in channel_dict["bottom_channels"]:
            index = channel_dict["bottom_channels"].index(ch)
            col = 'b' # cm.Blues(index/len(channel_dict["bottom_channels"]))
        elif ch in channel_dict["top_outer_ring"]:
            index = channel_dict["top_outer_ring"].index(ch)
            col = 'r' # cm.Reds(index/len(channel_dict["top_outer_ring"]))
        elif ch in channel_dict["top_bulk"]:
            index = channel_dict["top_bulk"].index(ch)
            col = 'g' #cm.Greens(index/len(channel_dict["top_bulk"]))

        if plot:
            plt.plot(actual_amplitude, actual_acceptance[ch], color=col, label='%i' % ch)
    if plot:
        plt.xlim(-2, 150)
        plt.ylim(0, 1.2)
        plt.grid(b=True)
        plt.xlabel('Amplitude threshold')
        plt.ylabel('SPE acceptance above threshold')
        plt.title("Acceptances from runs %d (Blue) %d (Red) %d (Green)" % (bottom_run, top_outer_run, top_bulk_run))
        plt.savefig(data_dir + "accepts_%d_%d_%d.png" % (bottom_run, top_outer_run, top_bulk_run))

    bottom_acc = get_acceptances(bottom_run, thresholds)
    top_outer_acc = get_acceptances(top_outer_run, thresholds)
    top_bulk_acc = get_acceptances(top_bulk_run, thresholds)

    actual_acceptance = np.ones(len(channels))

    low_acceptance_pmts = []

    for ch in channels:
        if ch in channel_dict["bottom_channels"]:
            actual_acceptance[ch] = bottom_acc[ch]
        elif ch in channel_dict["top_outer_ring"]:
            actual_acceptance[ch] = top_outer_acc[ch]
        elif ch in channel_dict["top_bulk"]:
            actual_acceptance[ch] = top_bulk_acc[ch]
        if actual_acceptance[ch] < 0.5 and ch not in excluded_pmts:
            low_acceptance_pmts.append(ch)

    print("Low acceptance pmts: ", low_acceptance_pmts)

    if plot:
        pmt_plot_file = data_dir + "pmtplot_%d_%d_%d.png" % (bottom_run, top_outer_run, top_bulk_run)
        plot_acceptances(actual_acceptance, pmt_plot_file)

    return actual_acceptance
    

def plot_acceptances(acceptances, output_file):
    # takes a list of acceptances, saves png to output_file

    pmtsizeArray = 700. * np.ones(len(acceptances))
    plot_on_pmt_arrays(color=acceptances,
                       size=pmtsizeArray,
                       geometry='physical',
                       colorbar_kwargs=dict(label='spe acceptance fraction'),
                       scatter_kwargs=dict(vmin=0.5, vmax=1))
    plt.suptitle('SPE acceptance fraction', fontsize=20)
    plt.savefig(output_file)


def write_to_txt(filename, bottom_run, top_bulk_run, top_outer_run):
    if not isinstance(bottom_run, int):
        bottom_run = int(bottom_run)

    if not isinstance(top_bulk_run, int):
        top_bulk_run = int(top_bulk_run)

    if not isinstance(top_outer_run, int):
        top_outer_run = int(top_outer_run)

    reg_run = find_regular_run(bottom_run)
    print("Using Run %d to get thresholds" % reg_run)
    thresholds = get_thresholds(reg_run)


    txtfiledir = data_dir_base.replace('data/', 'acceptances/')
    if not os.path.exists(txtfiledir):
        os.mkdir(txtfiledir)

    print("Getting acceptances...")
    acceptances = get_acceptances_3runs(bottom_run, top_outer_run, top_bulk_run, thresholds)
    print("Writing to %s" % txtfiledir + filename)
    with open(txtfiledir + filename, "w") as f:
        for ch, (thresh, acc) in enumerate(zip(thresholds, acceptances)):
            f.write("%d, %d, %0.3f \n" % (ch, thresh, acc))


if __name__ == '__main__':
    write_to_txt(*sys.argv[1:])
