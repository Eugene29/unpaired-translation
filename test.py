import sys
# sys.path.append("../")
from dataset import SplinterDataset
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pytest

import os
os.chdir('majorana')
from majorana import preprocess_h5py_file_into_nwfs
os.chdir('..')

if __name__ == '__main__':
    fname = 'MJD_NPML_0.hdf5'
    # with h5py.File(fname, 'r') as file:
    #     print(file.keys())
    # fname = 'MJD_Train_0.hdf5'
    # with h5py.File(fname, 'r') as file:
    #     print(file.keys())
    # print(file)
    # preprocessed_wfs = preprocess_h5py_file_into_nwfs(file)
    # print(preprocessed_wfs)


    # train_loader, test_loader = load_data(batch_size=16)

    def test_waveform_transformation():
        '''
        we want to make sure the our new, faster version of transforming h5py waveform with tp0 is correct.
        '''
        ## Get Label transformation ##
        dataset = SplinterDataset("DetectorPulses.pickle", "SimulatedPulses.pickle", debugging=True)
        gt_first_102 = np.concatenate([dataset[idx][0] for idx in np.arange(len(dataset))], axis=0)
        print(gt_first_102.shape)

        ## improved, general transformation ##
        import pickle
        fname = 'DetectorPulses.pickle'
        raw_waveform = []
        tp0 = []
        with open(fname, 'rb') as file:
            count = 0
            while True:
                try:
                    loaded = pickle.load(file)
                    raw_waveform.append(loaded['wf'][:1009]) ## match the dimension because og data is not homogeneous, changes the transformation formula a litte bit.
                    tp0.append(loaded['tp0'][0])
                    count += 1
                    if count > 101:
                        raise EOFError
                except EOFError:
                    break
        dict1 = {"raw_waveform": raw_waveform, "tp0": tp0}

        improved_wfs = preprocess_h5py_file_into_nwfs(dict1)
        print(improved_wfs.shape)
        assert 0.0 == (gt_first_102 - improved_wfs).sum()
        print((gt_first_102 - improved_wfs).sum())

    def plot_transformed_wf():
        ## visualization of one instance of the transformed wf using the new, faster pipeline. 
        plt.close()
        fname = "./majorana/MJD_Test_5.hdf5"
        with h5py.File(fname, 'r') as file:
            print(file['raw_waveform'].shape)
            transformed_wf = preprocess_h5py_file_into_nwfs(file)
            print(transformed_wf.shape)
            plt.plot(transformed_wf[0])
            plt.savefig("./majorana/transformed_wf.png")

    test_waveform_transformation()
    plot_transformed_wf()