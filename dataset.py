import numpy as np
import torch.utils.data as data_utils
import torch.nn as nn
import pickle
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import h5py
import sys
import os
sys.path.append('./majorana/')
from majorana import preprocess_h5py_file_into_nwfs


'''
Parameters for training waveform construction.
LSPAN: how many sample to select to the left of time point 0 (start of the rise)
RSPAN: how many sample to select to the right of time point 0 (start of the rise)
SEQ_LEN: total length of the input pulses, always equal to LSPAN+RSPAN
'''
LSPAN=300
RSPAN=500
SEQ_LEN=LSPAN+RSPAN

class SplinterDataset(Dataset):
    '''
    Splinter is the name of our local Ge detector
    '''

    def __init__(self, event_dset = "DetectorPulses.pickle", siggen_dset ="SimulatedPulses.pickle", debugging=False, debugging_w_nn=False):
        
        self.MJD = False
        self.debugging = debugging
        self.debugging_w_nn = debugging_w_nn
        siggen_dict = self.event_loader(siggen_dset)
        if isinstance(event_dset, list):
            self.MJD = True
            event_dict = self.h5py_loader(event_dset) ## Not a dictionary but full nwfs
        elif self.debugging_w_nn: ## to see if neural network trains the same when we use the new version of event loader for faster loaders (that also works with h5py)
            event_dict = self.event_loader_v2(event_dset)
        else:
            event_dict = self.event_loader(event_dset)
        self.siggen_dict = siggen_dict
        self.event_dict = event_dict
        self.size = len(self.event_dict)
        self.sim_size = len(self.siggen_dict)
        print(f"detector_size: {self.size}")

        self.plot_waveform(np.random.randint(self.size))
        
    def __len__(self):
        return self.size
    
    
    def transform(self,wf, tp0, MC=False):
        wf = np.array(wf)
        try:
            tp50=tp0[0]
        except:
            tp50 = tp0
        left_padding = max(LSPAN-tp50,0)
        right_padding = max((RSPAN+tp50)-len(wf),0)
        wf = np.pad(wf,(left_padding, right_padding),mode='edge')
        tp50 = tp50+left_padding
        wf = wf[(tp50-LSPAN):(tp50+RSPAN)]
        wf = (wf - wf.min())/(wf.max()-wf.min())
        return wf

    # @torchsnooper.snoop()
    def __getitem__(self, idx):
        #stack two waveforms together randomly
        # np.random.seed(idx)
        siggendict1 = self.siggen_dict[np.random.randint(self.sim_size)]
        siggendict2 = self.siggen_dict[np.random.randint(self.sim_size)]
        randflag = np.random.rand()
        # if randflag > 0.7:
        #     alpha = 1
        # elif randflag < 0.1:
        #     alpha = 511/(2615-511)
        # else:
        #     alpha = np.random.rand()
        alpha = 511/(2614.5-511)
        if randflag > 0.3:
            alpha = 1
        # elif randflag > 0.2:
        #     alpha = np.random.rand()
        siggenwf1 = self.transform(siggendict1["wf"],siggendict1["tp0"],MC=True)
        siggenwf2 = self.transform(siggendict2["wf"],siggendict2["tp0"],MC=True)
        siggenwf = (siggenwf1*alpha+siggenwf2*(1-alpha))

        if self.MJD or self.debugging_w_nn:
            return self.event_dict[idx][None, :], siggenwf[None,:], ["useless"]
        else:
            return self.transform(self.event_dict[idx]["wf"],self.event_dict[idx]["tp0"])[None,:], siggenwf[None,:], ["useless"]
        # return self.transform(self.event_dict[idx]["wf"],self.event_dict[idx]["tp0"])[None,:], siggenwf[None,:], self.event_dict[idx]["wf"][None,:SEQ_LEN]
        
    def return_label(self):
        return self.trainY
    
    def set_raw_waveform(self,raw_wf):
        self.raw_waveform = raw_wf

    def get_original_waveform(self,wf, input=False):
        if input:
            return self.input_transform.recon_waveform(wf)
        else:
            return self.output_transform.recon_waveform(wf)
    
    ## h5py loader
    def h5py_loader(self, fnames):
        # rel_path = "./majorana/"
        # for i, fname in enumerate(fnames):
        #     with h5py.File(rel_path+fname, 'r') as file:
        #         if i == 0:
        #             nwfs = preprocess_h5py_file_into_nwfs(file)
        #         else:
        #             nwfs = np.concatenate([nwfs, preprocess_h5py_file_into_nwfs(file)], axis=0)
        for i, fname in enumerate(fnames):
            with h5py.File(fname, 'r') as file:
                if "energy_label" not in file:
                    continue
                elif i==0:
                    preprocessed_wfs = preprocess_h5py_file_into_nwfs(file, energy_filtering=True)
                else:
                    preprocessed_wfs = np.concatenate([preprocessed_wfs, preprocess_h5py_file_into_nwfs(file, energy_filtering=True)], axis=0)
            print(len(preprocessed_wfs))
        return preprocessed_wfs
        
    #Load event from .pickle file
    def event_loader(self, address,elow=-99999,ehi=99999):
        wf_list = []
        ts_list = []
        count = 0
        with (open(address, "rb")) as openfile:
            while True:
                try:
                    wdict = pickle.load(openfile, encoding='latin1')
                    wf = wdict["wf"]
                    if "dc_label" in wdict.keys() and wdict["dc_label"] != 0.0:
                        continue
                    tp0 = wdict["tp0"]
                    try:
                        tp0=tp0[0]
                    except:
                        tp0 = tp0
                    nwf = (wf - wf.min())/(wf.max()-wf.min())
                    if np.nan in nwf:
                        continue
                    # if (self.pileup_cut(nwf)>7):
                    #     continue
                    # plt.plot(nwf[tp0:tp0+100])
                    if len(self.transform(wdict["wf"],wdict["tp0"],MC=True)) == SEQ_LEN:
                        wf_list.append(wdict)
                        count += 1
                    if self.debugging and count > 101:
                        break
                except EOFError:
                    break
        return wf_list
    
    def event_loader_v2(self, address,elow=-99999,ehi=99999):
        raw_waveform = []
        tp0 = []
        with open(address, 'rb') as file:
            count = 0
            while True:
                try:
                    loaded = pickle.load(file)
                    raw_waveform.append(loaded['wf'][:1009]) ## match the dimension because og data is not homogeneous, changes the transformation formula a litte bit.
                    tp0.append(loaded['tp0'][0])
                    count += 1
                    # if count > 101:
                    #     raise EOFError
                except EOFError:
                    break
        dict1 = {"raw_waveform": raw_waveform, "tp0": tp0}
        improved_wfs = preprocess_h5py_file_into_nwfs(dict1)
        return improved_wfs

    def get_field_from_dict(self, input_dict, fieldname):
        field_list = []
        for event in input_dict:
            field_list.append(event[fieldname])
        return field_list
    
    def get_current_amp(self,wf):
        return max(np.diff(wf.flatten()))
    
    def plot_waveform(self,idx):
        plt.figure(figsize=(15,15))
        plt.subplot(211)
        for i in range(100):
            waveform, waveform_deconv, rawwf = self.__getitem__(i)
            plt.plot(waveform[0],linewidth=0.5)
        plt.title("Smoothed Data")
        plt.xlabel("Time Sample")
        plt.ylabel("ADC counts")
        plt.subplot(212)
        for i in range(100):
            waveform, waveform_deconv, rawwf = self.__getitem__(i)
            plt.plot(waveform_deconv[0],linewidth=0.5)
        plt.title("Simulated WF")
        plt.xlabel("Time Sample")
        plt.ylabel("ADC counts")