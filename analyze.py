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
        val_to_check = [5, 6, 7, 8]
        big_array = np.ones((248, len(data['bin_centers']), len(val_to_check)))
        occupancy_array = np.ones((248, len(val_to_check)))
            
        for i, val in enumerate(val_to_check):
            corr, sigma_corr=self.make_correction(val, 'amplitude')
            big_array[: , : , i] = self.acceptance(val, 'amplitude', errors=False)
            occupancy_array[:,i] = -1*np.log(corr)
        # with systematics
        self.big_array = big_array
        self.occupancy_by_channel = (np.mean(occupancy_array, axis=1), np.std(occupancy_array, axis=1))
        self.off_channels = np.where(self.occupancy_by_channel[0] < 0.05)[0]
        systematics =  np.std(big_array, axis=2)
        accs, statistical = self.acceptance(6)
        self.acceptance_by_channel = accs, statistical, systematics
        

    def make_correction(self, val2corr2, space):
        if space not in ['amplitude', 'charge']:
            raise ValueError('must specify amplitude or charge')
        led = np.array(self.data['LED_%s' % space].copy())
        noise = np.array(self.data['noise_%s' % space].copy())
        sigma_led=np.sqrt(led)
        sigma_noise=np.sqrt(noise)
        bin2corr2 = np.where(self.data['bin_centers'] == val2corr2 + 0.5)[0][0]
        led_firstN = led[:bin2corr2, :].sum(axis=0)
        noise_firstN = noise[:bin2corr2,:].sum(axis=0)
        sigma_led_firstN=np.sqrt(np.sum(sigma_led[:bin2corr2, :]**2, axis=0))
        sigma_noise_firstN=np.sqrt(np.sum(sigma_noise[:bin2corr2,:]**2, axis=0))
        
        corr=led_firstN / noise_firstN
        sigma_corr=corr*np.sqrt((sigma_led_firstN/led_firstN)**2 + (sigma_noise_firstN/noise_firstN)**2)
        return corr, sigma_corr


    def residual(self, val2corr2, space='amplitude'):
        # subtract noise spectra from LED spectra for all channels
        # correct noise spectra by forcing sum up to val=x to be equal for both noise, led
        corrections, sigma_corr = self.make_correction(val2corr2, space)
        
        led = self.data['LED_%s' % space].copy()
        noise = self.data['noise_%s' % space].copy()
        sigma_led=np.sqrt(led)
        sigma_noise=np.sqrt(noise)
        corr_noise=noise * corrections
        sigma_corr_noise=corr_noise*np.sqrt( (sigma_corr/corrections)**2 + (sigma_noise/noise)**2)
        
        # return transpose so that is subscriptable by channel number
        res=(led - corr_noise).T
        sigma_res=np.sqrt( sigma_led**2 + sigma_corr_noise**2).T
       
        return res, sigma_res
     
    def acceptance(self, val2corr2, space='amplitude', errors=True):
        residual, residual_err = self.residual(val2corr2, space)
        acc = A_func(residual)
        
        if errors:
            minus1 = np.zeros_like(residual)
            plus1 = np.zeros_like(minus1)
            
            for ch in np.arange(248):
                MCs = acc_MC(residual[ch], residual_err[ch], 100)
                
                minus1[ch, :] = np.percentile(MCs, 16, axis=0)
                plus1[ch, :] = np.percentile(MCs, 84, axis=0)
                    
            errors = np.array([minus1, plus1])
            return acc, errors
        return acc

def A_func(residual):
    return 1 - residual.cumsum(axis=1) / residual.sum(axis=1)[:, np.newaxis]

def acc_MC(residual, sigma_res, total_curves):
    new_res=np.zeros((total_curves, len(residual)), dtype=float)
    new_accs=np.zeros((total_curves, len(residual)), dtype=float)
    samples=np.zeros((len(residual), total_curves), dtype=float)
    for b, res in enumerate(residual):
        mu=res
        sigma=abs(sigma_res[b])
        if np.isnan(sigma):
            new_res[:,b]=mu
        else:
            sample=norm.rvs(loc=mu, scale=sigma, size=total_curves)
            samples[b]=sample
            
        new_res=np.transpose(samples)
            
    new_accs =  A_func(new_res)
    #clip to prevent unphysical values
    new_acc=np.clip(new_accs, -0.1, 1.1)
    
    return new_acc
                                                                                                                                                            
class ch_data:
    def __init__(self, runlist, date, acc, on_acc, acc_errs_l, acc_errs_u, acc_sys, acc_stat, occ, on_occ, occ_sys, occ_stat):
        self.runlist=runlist
        self.date=date
        
        #all of the following are subscriptable by channel
        self.acc=acc
        self.on_acc=on_acc
        self.acc_errs_l=acc_errs_l
        self.acc_errs_u=acc_errs_u
        self.acc_sys=acc_sys
        self.acc_stat=acc_stat
        self.occ=occ
        self.on_occ=on_occ
        self.occ_sys=occ_sys
        self.occ_stat=occ_stat
        
        #self._on_channels = np.where(self.occ > 0.05)[0]
        
        

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
    thresholds = np.array(thresholds)[:248]
    bin0 = np.where(s.data['bin_centers'] == 0.5)[0][0]

    acc, stat, sys = s.acceptance_by_channel
    index = np.arange(len(acc)), bin0+thresholds
    acc_frac = acc[index]
    stat_frac = np.absolute(acc_frac - np.array([stat[0, index[0], index[1]], stat[1, index[0], index[1]]]))
    sys_frac = sys[index] 
    return acc_frac, stat_frac, sys_frac

def acceptance_3runs(bottom_run, topbulk_run, topring_run, thresholds):
    thresholds = np.array(thresholds)[:248]
    ret_acc, ret_stat_errs, ret_sys_errs = np.ones(248), np.ones((2, 248)), np.ones(248)
    run_list = [bottom_run, topbulk_run, topring_run]
    
    channel_lists = [channel_dict['bottom_channels'],
                     channel_dict['top_bulk'],
                     channel_dict['top_outer_ring']]

    for run, ch_list in zip(run_list, channel_lists):
        frac,  stat_errs, sys_errs = acceptance_fraction(run, thresholds)
        ret_acc[ch_list] = frac[ch_list]
        ret_stat_errs[:, ch_list] = stat_errs[:, ch_list]
        ret_sys_errs[ch_list] = sys_errs[ch_list]
        
    return ret_acc, ret_stat_errs, ret_sys_errs
    
def acceptance_curve_3runs(bottom_run, topbulk_run, topring_run):
    ret_acc, ret_errs= np.ones((248, 1099)), np.ones((248, 1099))
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
        ret_errs[ch_list,:] = errs[ch_list,:]
        ret_errs=errs
        x = s.data['bin_centers']
    return x, ret_acc , ret_errs

#add stats errors to occ, use log of corr
def occupancy(run_number):
    path = os.path.join(data_dir_base, 'run_%05d.h5' % run_number)
    if not os.path.exists(path):
        print("Acceptance data does not exist for run %d" % run_number)
    s = SPE(path)
    occ, occ_sys=s.occupancy_by_channel
    corr, sigma_corr=s.make_correction(val2corr2=6, space='amplitude')
    occ_stat=-np.log(sigma_corr)
    return occ, occ_sys, occ_stat

def occupancy_3runs(bottom_run, topbulk_run, topring_run):
    ret_occ, ret_sys_errs, ret_stat_errs = np.ones(248), np.ones(248), np.ones(248)
    run_list = [bottom_run, topbulk_run, topring_run]
    channel_lists = [channel_dict['bottom_channels'],
                     channel_dict['top_bulk'],
                     channel_dict['top_outer_ring']]
    for run, ch_list in zip(run_list, channel_lists):
        occ, occ_sys, occ_stat = occupancy(run)
        ret_occ[ch_list] = occ[ch_list]
        ret_sys_errs[ch_list] = occ_sys[ch_list]
        ret_stat_errs[ch_list] = occ_stat[ch_list]
    return ret_occ, ret_sys_errs, ret_stat_errs


def twoplus_contribution(occ):
    return 1 - np.exp(-occ)*(1+occ)

    
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


