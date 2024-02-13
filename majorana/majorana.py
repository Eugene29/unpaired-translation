from bs4 import BeautifulSoup
import requests
import glob

## get all .hdf5 data from zenodo##
# url = "https://zenodo.org/records/8257027"

# response = requests.get(url)
# response.raise_for_status

# soup = BeautifulSoup(response.text, "html.parser")
# links = soup.find_all('link', {'rel':"alternate", 'type':"application/octet-stream"})
# urls = [link["href"] for link in links]

# print(urls)

# for url in urls:
#     fname = url.split('/')[-1]
#     with requests.get(url, stream=True) as r:
#         r.raise_for_status()
#         with open(fname, 'wb') as f:
#             for chunk in r.iter_content(chunk_size=8192):
#                 f.write(chunk)
#     print(f"Downloaded {fname}")
## ------------------- ##
    

## Open all .hdf5 files ##
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pickle

# with (open('../DetectorPulses.pickle', "rb")) as openfile:
    # wdict = pickle.load (openfile, encoding='latin1')
    # print(wdict['wf'].shape)
LSPAN = 300
RSPAN = 500

def preprocess_h5py_file_into_nwfs(file, energy_filtering=False):
    low_energy = 2094
    high_energy = 2104
    wfs = np.array(file["raw_waveform"], dtype=np.float32)
    tp0s = np.array(file["tp0"])
    if energy_filtering:
        energy = np.array(file['energy_label'])
        wfs = wfs[(low_energy < energy) & (energy < high_energy)]
        tp0s = tp0s[(low_energy < energy) & (energy < high_energy)]
    fil_wfs = wfs[(tp0s >= LSPAN) & (tp0s <= wfs.shape[-1] - RSPAN)]
    fil_tp0s = tp0s[(tp0s >= LSPAN) & (tp0s <= wfs.shape[-1] - RSPAN)]
    row_indicies = np.arange(len(fil_wfs))[:, np.newaxis] ## adds extra dim
    col_indicies = np.array([np.arange(tp0-LSPAN, tp0+RSPAN) for tp0 in fil_tp0s]) ## super advanced indexing
    tp0_fil_wfs = fil_wfs[row_indicies, col_indicies]
    tp0_fil_wfs = normalize_all(tp0_fil_wfs)

    not_selected_wfs = wfs[(tp0s < LSPAN) | (tp0s > wfs.shape[-1] - RSPAN)]
    not_selected_tp0s = tp0s[(tp0s < LSPAN) | (tp0s > wfs.shape[-1] - RSPAN)]
    for wf, tp0 in zip(not_selected_wfs, not_selected_tp0s):
        # print(transform2(wf, tp0).shape)
        tp0_fil_wfs = np.concatenate([tp0_fil_wfs, transform2(wf, tp0)[None, :]], axis=0) ## [N, Seq_len]
    ## check nans
    if np.any(np.isnan(tp0_fil_wfs)):
        raise ValueError
    return tp0_fil_wfs
    
def transform2(wf, tp0, MC=False):
    left_padding = max(LSPAN-tp0,0)
    right_padding = max((RSPAN+tp0)-len(wf),0)
    wf = np.pad(wf,(left_padding, right_padding),mode='edge')
    tp0 = tp0+left_padding
    wf = wf[(tp0-LSPAN):(tp0+RSPAN)]
    wf = (wf - wf.min())/(wf.max()-wf.min())
    return wf

def normalize_all(wfs):
    ## Normalize
    range_ = wfs.max(axis=-1) - wfs.min(axis=-1)
    no_negative = wfs - wfs.min(axis=-1)[:, None]
    nwfs = no_negative / range_[:, None]
    return nwfs

def preprocess_all_h5py_files_into_nwfs():
    # import os
    # directory = '/home/eku/cpu-gan/majorana/'
    # fname_lst = glob.glob(os.path.join(directory, "*.hdf5"))
    
    for i, fname in enumerate(fname_lst):
        with h5py.File(fname, 'r') as file:
            if "energy_label" not in file:
                continue
            elif i==0:
                preprocessed_wfs = preprocess_h5py_file_into_nwfs(file, energy_filtering=True)
            else:
                preprocessed_wfs = np.concatenate([preprocessed_wfs, preprocess_h5py_file_into_nwfs(file, energy_filtering=True)], axis=0)
        print(len(preprocessed_wfs))
    return preprocessed_wfs

if __name__ == '__main__':
    # fname = 'MJD_Train_0.hdf5'
    fname_lst = glob.glob("*.hdf5")
    for i, fname in enumerate(fname_lst):
        with h5py.File(fname, 'r') as file:
            if "energy_label" not in file:
                continue
            elif i==0:
                preprocessed_wfs = preprocess_h5py_file_into_nwfs(file, energy_filtering=True)
            else:
                preprocessed_wfs = np.concatenate([preprocessed_wfs, preprocess_h5py_file_into_nwfs(file, energy_filtering=True)], axis=0)
        print(len(preprocessed_wfs))
    # print(preprocessed_wfs)

    # import pickle
    # fname = '../DetectorPulses.pickle'
    # raw_waveform = []
    # tp0 = []
    # with open(fname, 'rb') as file:
    #     while True:
    #         try:
    #             loaded = pickle.load(file)
    #             raw_waveform.append(loaded['wf'])
    #             tp0.append(loaded['tp0'])
    #         except EOFError:
    #             break
    # train_loader, test_loader = load_data(batch_size=16)

    # dataset = SplinterDataset("DetectorPulses.pickle", "SimulatedPulses.pickle")
    # print(dataset)