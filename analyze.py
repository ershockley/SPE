import matplotlib.pyplot as plt
import re
import numpy as np
from pax.configuration import load_configuration
import hax
from hax.pmt_plot import plot_on_pmt_arrays
from channel_dict import channel_dict
import pandas as pd
import os
import runDB
import math
import scipy.stats as stats
from scipy.stats import norm


from spe_acceptance import data_dir_base


hax.init()

class SPE:
    def __init__(self, path):
        df = pd.read_hdf(path)
        data = {}
        data['bin_centers'] = df['bins'].values
        for key in ['LED_amplitude', 'LED_charge', 'noise_amplitude', 'noise_charge']:
            data[key] = np.array(list(df[key].values))
        self.data = data.copy()

        # numpy magic
        val_to_check = [4, 5, 6, 7, 8, 9, 10]
        big_array = np.ones((248, len(data['bin_centers']), len(val_to_check)))
        occupancy_array = np.ones((248, len(val_to_check)))
        for i, val in enumerate(val_to_check):
            corr, sigma_corr=self.make_correction(val, 'amplitude')
            big_array[:, :, i] = self.acceptance(val, 'amplitude')
            occupancy_array[:,i] = -1*np.log(corr)
        # with systematics
        self.big_array = big_array
        self.occupancy_by_channel = (np.mean(occupancy_array, axis=1), np.std(occupancy_array, axis=1))
        self.off_channels = np.where(self.occupancy_by_channel[0] < 0.05)[0]
        self.acceptance_by_channel = (np.mean(big_array, axis=2), np.std(big_array, axis=2))


    def make_correction(self, val2corr2, space):
        if space not in ['amplitude', 'charge']:
            raise ValueError('must specify amplitude or charge')
        led = np.array(self.data['LED_%s' % space].copy())
        noise = np.array(self.data['noise_%s' % space].copy())
        sigma_led=np.array([np.sqrt(abs(bin)) for bin in led])
        sigma_noise=np.array([np.sqrt(abs(bin)) for bin in noise])
        bin2corr2 = np.where(self.data['bin_centers'] == val2corr2 + 0.5)[0][0]
        led_firstN = led[:bin2corr2, :].sum(axis=0)
        noise_firstN = noise[:bin2corr2, :].sum(axis=0)
        corr=led_firstN / noise_firstN
        sigma_corr=corr*np.sqrt((sigma_led/led)**2 + (sigma_noise/noise)**2)
        return corr, sigma_corr


    def residual(self, val2corr2=6, space='amplitude'):
        # subtract noise spectra from LED spectra for all channels
        # correct noise spectra by forcing sum up to val=x to be equal for both noise, led
        corrections, sigma_corr = self.make_correction(val2corr2, space)
        led = self.data['LED_%s' % space].copy()
        noise = self.data['noise_%s' % space].copy()
        sigma_led=np.array([np.sqrt(abs(bin)) for bin in led])
        sigma_noise=np.array([np.sqrt(abs(bin)) for bin in noise])
        corr_noise=noise * corrections
        sigma_corr_noise=corr_noise*np.sqrt( (sigma_corr/corrections)**2 + (sigma_noise/noise)**2)
        # return transpose so that is subscriptable by channel number
        res=(led - noise).T
        sigma_res=np.array(np.sqrt( sigma_led**2 + sigma_corr_noise**2)).T
        return res, sigma_res
     
    #make lots of acc curves using numpy magic
    def acc_MC(self, residual, sigma_res, total_curves):
        new_res=np.zeros((total_curves, len(residual)), dtype=float)
        new_accs=np.zeros((total_curves, len(residual)), dtype=float)
        samples=np.zeros((len(residual), total_curves), dtype=float)
        for b, res in enumerate(residual):
            mu=res
            sigma=abs(sigma_res[b])
            if math.isnan(sigma)==True:
                new_res[:,b]=mu
            else:
                x=np.linspace(mu-3*sigma, mu+3*sigma, num=1000)
                sample=norm.rvs(loc=mu, scale=sigma, size=total_curves)
                samples[b]=sample
            new_res=np.transpose(samples)
        cs=np.cumsum(new_res, axis=1) 
        s=np.sum(new_res, axis=1)
        ns=s[:,np.newaxis] 
        ratio=cs/ns 
        ones=np.ones( (total_curves, len(residual))) 
        new_accs=np.subtract(ones, ratio)
        new_accs=np.clip(new_accs, -0.1, 1.1)  
        return new_accs
        

    def acceptance(self, val2corr2=6, space='amplitude'):
        residual, sigma_res = self.residual(val2corr2, space)
        acc =  1 - residual.cumsum(axis=1) / residual.sum(axis=1)[:, np.newaxis]
        #clip to prevent unphysical values
        acc=np.clip(acc, -0.1, 1.1)
        return acc



############################################################################################

def get_thresholds(run_number):
    # thanks Jelle
    pax_config = load_configuration('XENON1T')
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

    return [thresholds.get(i, default_threshold) for i in np.arange(len(lookup_pmt))]

def acceptance_fraction(run_number, thresholds):
    path = os.path.join(data_dir_base, 'run_%05d.h5' % run_number)
    if not os.path.exists(path):
        print("Acceptance data does not exist for run %d" % run_number)
    s = SPE(path)
    res, sigma_res=s.residual(6, 'amplitude')
    frac_array = np.ones((248, s.big_array.shape[-1]))#, len(residual(axis=1)))
    ch_index = np.arange(248)
    thresholds = np.array(thresholds)[:248]
    bin0 = np.where(s.data['bin_centers'] == 0.5)[0][0]
    for i in range(s.big_array.shape[-1]):
        frac_array[:,i] = s.big_array[...,i][ch_index, thresholds + bin0]
    acc_frac = np.mean(frac_array, axis=1)
    sys_errs = np.std(frac_array, axis=1)
    acc_frac[s.off_channels] = 0
    sys_errs[s.off_channels] = 0
    
    #use MC to find statistical errors
    #acc_errs=np.empty( (248, len(s.data['bin_centers']), len(s.data['bin_centers'])) )
    sigma_l=np.zeros((248, len(s.data['bin_centers'])))
    sigma_u=np.zeros((248, len(s.data['bin_centers'])))
    acc_errs_l=np.zeros((248, len(s.data['bin_centers'])))
    acc_errs_u=np.zeros((248, len(s.data['bin_centers'])))
    for ch in ch_index:
        s=SPE(path)
        acc_curves=s.acc_MC(res[ch], sigma_res[ch], 100)
        
        sigma_l[ch,:]=np.percentile(acc_curves,16, axis=0)-np.mean(acc_curves, axis=0)
        sigma_u[ch,:]=np.percentile(acc_curves, 84, axis=0)-np.mean(acc_curves, axis=0)
        stat_errs=np.array([sigma_l, sigma_u])
        
        acc_errs_u[ch]=np.sqrt(sys_errs[ch]**2+np.mean(sigma_l, axis=0)**2)
        acc_errs_l[ch]=np.sqrt(sys_errs[ch]**2+np.mean(sigma_u, axis=0)**2)
    
    acc_errs=np.array( (np.array(ch_index), np.array(acc_errs_l[1]), np.array(acc_errs_u[1])))
    #for ch1, ch2 in zip(acc_errs_l[0], acc_errs_u[0]):
     #   acc_errs[0]=ch
      #  acc_errs[ch1]=acc_errs_l[ch1]
       # acc_errs[ch2]=acc_errs_u[ch2]
    #np.append(acc_errs, acc_errs_l[1], axis=1)
    #np.append(acc_errs, acc_errs_u[1], axis=2)
                         
    return acc_frac, acc_errs


def acceptance_3runs(bottom_run, topbulk_run, topring_run, thresholds):
    thresholds = np.array(thresholds)[:248]
    ret_acc, ret_errs_l, ret_errs_u = np.ones(248), np.ones(248), np.ones(248)
    run_list = [bottom_run, topbulk_run, topring_run]
    channel_lists = [channel_dict['bottom_channels'],
                     channel_dict['top_bulk'],
                     channel_dict['top_outer_ring']]
    for run, ch_list in zip(run_list, channel_lists):
        frac, errs = acceptance_fraction(run, thresholds)
        ret_acc[ch_list] = frac[ch_list]
        ret_errs_l[ch_list] = errs[1][ch_list]
        ret_errs_u[ch_list] = errs[2][ch_list]
    ret_errs=[ret_errs_l, ret_errs_u]
    return ret_acc, ret_errs

def acceptance_curve_3runs(bottom_run, topbulk_run, topring_run):
    ret_acc, ret_errs_l, ret_errs_u = np.ones((248, 1099)), np.ones((248, 1099)), np.ones((248, 1099))
    run_list = [bottom_run, topbulk_run, topring_run]
    channel_lists = [channel_dict['bottom_channels'],
                     channel_dict['top_bulk'],
                     channel_dict['top_outer_ring']]
    for run, ch_list in zip(run_list, channel_lists):
        path = os.path.join(data_dir_base, 'run_%05d.h5' % run)
        if not os.path.exists(path):
            print("Acceptance data does not exist for run %d" % run)
        s = SPE(path)
        frac, errs = s.acceptance_by_channel
        ret_acc[ch_list,:] = frac[ch_list,:]
        ret_errs_l[ch_list,:] = errs[1][ch_list,:]
        ret_errs_l[ch_list,:] = errs[2][ch_list,:]
        ret_errs=[ret_errs_l, ret_errs_u]
        x = s.data['bin_centers']
    return x, ret_acc, ret_errs

def occupancy(run_number):
    path = os.path.join(data_dir_base, 'run_%05d.h5' % run_number)
    if not os.path.exists(path):
        print("Acceptance data does not exist for run %d" % run_number)
    s = SPE(path)
    return s.occupancy_by_channel

def occupancy_3runs(bottom_run, topbulk_run, topring_run):
    ret_occ, ret_errs = np.ones(248), np.ones(248)
    run_list = [bottom_run, topbulk_run, topring_run]
    channel_lists = [channel_dict['bottom_channels'],
                     channel_dict['top_bulk'],
                     channel_dict['top_outer_ring']]
    for run, ch_list in zip(run_list, channel_lists):
        occ, errs = occupancy(run)
        ret_occ[ch_list] = occ[ch_list]
        ret_errs[ch_list] = errs[ch_list]
    return ret_occ, ret_errs

def plot_channel(ch, run_number, xlims, ylims = (-100, 500), filedir = ''):
    return
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


def twoplus_contribution(occ):
    return 1 - np.exp(-occ)*(1+occ)


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


def find_threshold(bin_centers, acceptance, acc_frac):
    next_a, next_b = (-99, -99)  # inital nonsense values
    thresh = 0

    for a, b in zip(reversed(acceptance), reversed(bin_centers-0.5)):
        if (a >= acc_frac >= next_a):
            if (abs(a - acc_frac) < abs(next_a - acc_frac)):
                thresh = b
            else:
                thresh = next_b
            break
        next_a = a
        next_b = b

    return int(thresh)


def find_regular_run(LED_run):
    collection = runDB.get_collection()
    query = {'source.type': {'$ne': 'LED'},
             '$and': [{'number': {'$lt': LED_run + 20}},
                      {'number': {'$gt': LED_run - 20}}
                      ]
             }
    cursor = collection.find(query, {'number': True,
                                     'reader': True,
                                     '_id': False})

    runs = np.array([run['number'] for run in cursor
                     if any([r['register'] == '8060'
                             for r in run['reader']['ini']['registers']])])

    if LED_run < 5144:  # when thresholds changed
        runs = runs[runs < 5144]
    elif LED_run > 5144:
        runs = runs[runs > 5144]
    diff = abs(runs - LED_run)

    closest_run = runs[np.argmin(diff)]

    return closest_run


