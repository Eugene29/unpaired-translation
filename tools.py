import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import torch
import torch.nn as nn
import torch.autograd as autograd
import time
import numpy as np
from tqdm import tqdm
from scipy.stats import norm
from dataset import SplinterDataset, SEQ_LEN, LSPAN, RSPAN
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torch.utils.data as data_utils
import matplotlib.pyplot as plt

def asym_trap_filter(w_in, rise, flat, fall):
    """
    Apply an asymmetric trapezoidal filter to the waveform, normalized
    by the number of samples averaged in the rise and fall sections.
    Parameters
    ----------
    w_in : array-like
        The input waveform
    rise : int
        The number of samples averaged in the rise section
    flat : int
        The delay between the rise and fall sections
    fall : int
        The number of samples averaged in the fall section
    w_out : array-like
        The normalized, filtered waveform
    Examples
    --------
    .. code-block :: json
        "wf_af": {
            "function": "asym_trap_filter",
            "module": "pygama.dsp.processors",
            "args": ["wf_pz", "128*ns", "64*ns", "2*us", "wf_af"],
            "unit": "ADC",
            "prereqs": ["wf_pz"]
        }
    """
    w_out = np.array([np.nan]*len(w_in))
    w_in = (w_in-w_in.min())/(w_in.max()-w_in.min())
    if np.isnan(w_in).any() or np.isnan(rise) or np.isnan(flat) or np.isnan(fall):
        return

    w_out[0] = w_in[0] / rise
    for i in range(1, rise, 1):
        w_out[i] = w_out[i-1] + w_in[i] / rise
    for i in range(rise, rise + flat, 1):
        w_out[i] = w_out[i-1] + (w_in[i] - w_in[i-rise]) / rise
    for i in range(rise + flat, rise + flat + fall, 1):
        w_out[i] = w_out[i-1] + (w_in[i] - w_in[i-rise]) / rise - w_in[i-rise-flat] / fall
    for i in range(rise + flat + fall, len(w_in), 1):
        w_out[i] = w_out[i-1] + (w_in[i] - w_in[i-rise]) / rise - (w_in[i-rise-flat] - w_in[i-rise-flat-fall]) / fall
    return w_out

def get_roc(sig,bkg):
    '''
    This function gets the false positive rate, true positive rate, cutting threshold 
    and area under curve using the given signal and background array
    '''
    testY = np.array([1]*len(sig) + [0]*len(bkg))
    predY = np.array(sig+bkg)
    auc = roc_auc_score(testY, predY)
    fpr, tpr, thr = roc_curve(testY, predY)
    return fpr,tpr,thr,auc


# def get_tail_slope(wf):
#     '''
#     This function calculates the tail slope of input waveform
#     '''
#     premax_wf = wf[:wf.argmax()]
#     point97 = np.argmin(np.abs(premax_wf - 0.97))
#     last_pt = point97+200
#     first_occurence = np.mean(wf[(last_pt-50):(last_pt)])
#     last_occurence = np.mean(wf[-100:-50])
#     return (last_occurence-first_occurence)/(len(wf)-50-last_pt)


# def get_ca(wf):
#     '''
#     This function calculates the current amplitude of input waveform
#     using a sliding window linear fit
#     '''
#     window = 10
#     dtslope = []
#     for cur_index in range(len(wf)-10):
#         x = np.arange(cur_index,cur_index+window,1)
#         y = wf[cur_index:cur_index+window]
#         # # print(x.shape,y.shape)
#         dtslope.append(np.polyfit(x,y,1)[0])
#         # dtslope.append((wf[cur_index+window]-wf[cur_index])/10)
#     return np.max(dtslope)

def get_ca(wf, DEVICE):
    '''
    This function calculates the current amplitude of input waveform
    using a sliding window linear fit, vectorized for efficiency.
    '''
    window = 10
    wf = wf.squeeze()
    num_coeff = len(wf) - window

    # Create indices for all sliding windows at once
    indices = torch.arange(num_coeff).unsqueeze(-1) + torch.arange(window)
    # Use advanced indexing to create the y values for all windows
    y = wf[indices].to(DEVICE)

    # Create the X matrix for all windows at once
    #### [IMPORTANT] chatgpt figured out that having the same x-axis will produce the same output for linear_regression.

    x = torch.arange(window, dtype=torch.float32)
    X = torch.vstack((x, torch.ones(window))).float().T
    X = X.unsqueeze(0).repeat(num_coeff, 1, 1).to(DEVICE)  # Repeat X for all windows

    # Perform batched least squares regression
    # We can't use torch.linalg.lstsq in a batched manner, so we do it manually
    XtX = X.transpose(1, 2) @ X
    Xty = X.transpose(1, 2) @ y.unsqueeze(2)
    XtX_inv = torch.linalg.inv(XtX)
    slopes = XtX_inv @ Xty

    # The slope is the first coefficient
    slopes = slopes[:, 0].squeeze(1)

    return slopes.max()

def get_tail_slope(wf, DEVICE):
    '''
    This function calculates the tail slope of input waveform
    '''
    tbc_wf = wf.squeeze()
    premax_wf = tbc_wf[:tbc_wf[:].argmax(axis=-1)]
    point97 = (premax_wf - 0.97).abs().argmin() ## idx
    last_pt = point97+200
    first_occurence = torch.mean(tbc_wf[(last_pt-50):(last_pt)])
    last_occurence = torch.mean(tbc_wf[-100:-50])
    return (last_occurence-first_occurence)/(tbc_wf.size(-1)-50-last_pt)

def compute_normalized_tail_slope(ATN, test_loader, DEVICE, deterministic=False, accelerator=None):
    start = time.time()
    if accelerator is not None:
        accelerator.print('[Slow code]: Obtain the critical reconstruction parameters of each waveform by looping through the test dataset')
    ts = []
    gan_ts = []
    ca = []
    gan_ca = []
    sim_ca = []
    for wf, wf_deconv, rawwf in tqdm(test_loader):
        bsize = wf.size(0)
        gan_wf = ATN(wf_deconv.float().to(DEVICE), deterministic) ## Bug netG_B2A --> ATN
        for iwf in range(bsize):
            datawf = wf[iwf]
            siggenwf = wf_deconv[iwf]
            transfer_wf = gan_wf[iwf]
            ts.append(get_tail_slope(datawf, DEVICE).cpu().numpy())
            gan_ts.append(get_tail_slope(transfer_wf, DEVICE).cpu().numpy())
            ca.append(get_ca(datawf, DEVICE).cpu().numpy())
            gan_ca.append(get_ca(transfer_wf, DEVICE).cpu().numpy())
            sim_ca.append(get_ca(siggenwf, DEVICE).cpu().numpy())
        
    ## Aobo's og computation method for comparison            
#     for wf, wf_deconv, rawwf in tqdm(test_loader):
#         bsize = wf.size(0)
#         gan_wf = ATN(wf_deconv.to(DEVICE).float())
#         for iwf in range(bsize):
#             datawf = wf[iwf,0].cpu().numpy().flatten()
#             siggenwf = wf_deconv[iwf,0].cpu().numpy().flatten()
#             transfer_wf = gan_wf[iwf,0].detach().cpu().numpy().flatten()
#             ts.append(get_tail_slope(datawf))
#             gan_ts.append(get_tail_slope(transfer_wf))
#             ca.append(get_ca(datawf))
#             gan_ca.append(get_ca(transfer_wf))
#             sim_ca.append(get_ca(siggenwf))
    end = time.time()
    if accelerator is not None:
        accelerator.print(f"time: {end - start}") 
    return ts, gan_ts, ca, gan_ca, sim_ca

def normalize_tail_slope(ts, gan_ts):
    ts = np.array(ts)
    gan_ts = np.array(gan_ts)
    mean,std = norm.fit(select_quantile(ts))
    gan_mean, gan_std = norm.fit(select_quantile(gan_ts))
    return (np.array(ts)-mean)/std, (np.array(gan_ts)-gan_mean)/gan_std

def calc_gradient_penalty(netD, real_data, fake_data):
    '''
    This function calculates the gradient penalty of GAN-based model (ArXiv: 1704.00028)
    The idea is to apply 1-Lipshitz constratin on the latent space
    '''
    alpha = torch.rand(BATCH_SIZE, 1,1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(DEVICE)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.to(DEVICE)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(DEVICE),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()*10
    return gradient_penalty

def select_quantile(x):
    '''
    Select only the 10% to 90% quantile of the data
    used to calculate a more robust mean/std of long-tailed dataset
    '''
    quantilelow = np.quantile(x,0.10)
    quantilehi = np.quantile(x,0.90)
    return x[(x>quantilelow) & (x<quantilehi)]

def num_paramters(m):
    learnable = [p.numel() for p in m.parameters() if p.requires_grad]
    return sum(learnable)

def calculate_iou(h1,h2, rg, normed=False):
    '''
    Calculate the histogram intersection over union
    '''
    h1 = np.array(h1)
    h2 = np.array(h2)
    if normed:
        mean,std = norm.fit(select_quantile(h1))
        h1 = (h1-mean)/std
        mean,std = norm.fit(select_quantile(h2))
        h2 = (h2-mean)/std
    count, _ = np.histogram(h1,bins=rg,density=True)
    count2, _ = np.histogram(h2,bins=rg,density=True)
    intersection = 0
    union = 0
    for i in range(len(count)):
        intersection += min(count[i],count2[i])
        union += max(count[i],count2[i])
    return intersection/union*100.0

def inf_train_gen(train_loader):
    '''
    Allow us to sample infinitely (with repetition) from the training dataset
    '''
    while True:
        for wf, wf_deconv,rawwf in train_loader:
            yield wf, wf_deconv
            
def load_data(batch_size, MJD=False, validation_split=.3):
    if MJD:
        import glob
        import os
        # fnames = ["MJD_Test_0.hdf5"] ## change this to all fnames later.
        fnames = glob.glob(os.path.join('majorana', "*.hdf5")) ## all fnames
        dataset = SplinterDataset(event_dset=fnames, siggen_dset="SimulatedPulses.pickle")
    else:
        dataset = SplinterDataset("DetectorPulses.pickle", "SimulatedPulses.pickle")

    shuffle_dataset = True
    random_seed= 42222
    indices = np.arange(len(dataset))

    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    split = int(validation_split*len(dataset))
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    dataset.set_raw_waveform(False)
    train_loader = data_utils.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler,  drop_last=True)
    test_loader = data_utils.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler,  drop_last=True)

    return train_loader,test_loader

def plot_any_output(outputs, accelerator, save_dir):
    # from accelerate import Acceleartor
    # - Read a single batch from the test loader, translating it through the ATN
    accelerator.print('Read a single batch from the test loader, translating it through the ATN')
    wf, wf_deconv, _ = next(iter(outputs))
    wf = wf
    wf_deconv = wf_deconv

    outputs
    iwf = 2 # the ith waveform in the batch to plot
    detector_pulse = wf[iwf,0,:].cpu().data.numpy().flatten()
    simulated_pulse = wf_deconv[iwf,0,:].cpu().data.numpy().flatten()
    translated_pulse = outputs[iwf,0,:].cpu().data.numpy().flatten()

    # - Plot simulated pulses, data pulses and translated pulses in the same plot.
    accelerator.print('Plot simulated pulses, data pulses and translated pulses in the same plot.')
    fig = plt.figure(figsize=(25, 7))
    plt.plot(detector_pulse, label="Data Pulse",alpha=0.3, color="magenta", linestyle=":",linewidth = 4)
    plt.plot(simulated_pulse, label="Simulated Pulse",alpha=0.7, color="red", linewidth = 3)
    plt.plot(translated_pulse, label="ATN Output",color="dodgerblue", linewidth = 2)
    plt.axvspan(xmin=300,xmax=358,alpha=0.2,color="grey", label="Preamp Integration")
    plt.axvspan(xmin=358,xmax=800,alpha=0.1,color="cyan",label="RC Discharge")
    plt.xticks([], [])
    plt.yticks([], [])
    plt.xlabel("Time Sample [ns]")
    plt.ylabel("ADC Counts [a.u.]")
    # ax_main.plot(orwf, label="Data->Siggen",alpha=0.3,color="green", linewidth = 5)
    # plt.gca().get_xaxis().set_visible(False)
    # plt.gca().get_yaxis().set_visible(False)
    plt.legend(loc="upper left")
    plt.xlim(200, 600)
    plt.savefig(f"./{save_dir}/" + save_dir+"_ATN.png",dpi=200)
    plt.show()
    plt.cla()
    plt.clf()
    plt.close()

    # - Obtain the critical reconstruction parameters of each waveform by looping through the test dataset
    #     - `ca`: maximal current amplitude
    #     - `ts`: tail slope
    # - Note that this code is slow, mainly because of the current amplitdue calculation
    # ts, gan_ts, ca, gan_ca, sim_ca = compute_normalized_tail_slope(ATN, test_loader, DEVICE, deterministic=deterministic, accelerator=accelerator)

    ts = []
    gan_ts = []
    ca = []
    gan_ca = []
    sim_ca = []
    print(outputs.shape)
    # for wf, wf_deconv, rawwf in tqdm(test_loader):
    #     bsize = wf.size(0)
    #     gan_wf = ATN(wf_deconv.float().to(DEVICE), deterministic) ## Bug netG_B2A --> ATN
    #     for iwf in range(bsize):
    #         datawf = wf[iwf]
    #         siggenwf = wf_deconv[iwf]
    #         transfer_wf = gan_wf[iwf]
    #         ts.append(get_tail_slope(datawf, DEVICE).cpu().numpy())
    #         gan_ts.append(get_tail_slope(transfer_wf, DEVICE).cpu().numpy())
    #         ca.append(get_ca(datawf, DEVICE).cpu().numpy())
    #         gan_ca.append(get_ca(transfer_wf, DEVICE).cpu().numpy())
    #         sim_ca.append(get_ca(siggenwf, DEVICE).cpu().numpy())

    # n_ts, n_gan_ts = normalize_tail_slope(ts, gan_ts)
    
    # # - Plotting the normalized tail slope
    # fig = plt.figure(figsize=(10,8))
    # plt.rcParams['font.size'] = 20
    # plt.rcParams["figure.figsize"] = (10,8)
    # rg = np.linspace(-4,16,50)
    # plt.hist(n_ts,bins=rg,histtype="step",linewidth=2,density=False,color="dodgerblue",label="Detector Pulse")
    # plt.hist(n_gan_ts,bins=rg,histtype="step",linewidth=2,density=False,color="magenta",label="ATN Output Pulse")
    # plt.axvline(x=0,color="deeppink",linewidth=3,label="Simulated Pulse")
    # plt.legend()
    # plt.ylabel("# of Waveforms/ 0.02 [a.u.]")
    # plt.xlabel("Normalized Tail Slope [a.u.]")
    # plt.savefig(f"./{save_dir}/" + save_dir + "_tailslope.png",dpi=100)
    # print("normalized slope_HIoU:")
    # slope_HIoU = calculate_iou(ts, gan_ts, rg=rg, normed=True)
    # print(slope_HIoU)
    # accelerator.log({"slope_HIoU": slope_HIoU})
    # print("-------")
    
    # # - Plotting the maximal current amplitude
    # fig = plt.figure(figsize=(10,8))
    # plt.rcParams['font.size'] = 20
    # plt.rcParams["figure.figsize"] = (10,8)
    # rg = np.linspace(0.05,0.12,50)
    # plt.hist(gan_ca,bins=rg,label="ATN Output Pulse",alpha=0.1,color="magenta")
    # plt.hist(sim_ca,bins=rg,label="Simulated Pulse",linewidth=2,histtype="step",color="deeppink")
    # plt.hist(ca,bins=rg,label="Detector Pulse",histtype="step",linewidth=2,color="dodgerblue")
    # plt.xlabel("Current Amplitude [Normalized ADC Count / 100 ns]")
    # plt.ylabel("# of Events / 0.001 Current Amplitude")
    # plt.legend(loc="upper left")
    # # plt.yscale("log")
    # plt.savefig(f"./{save_dir}/"+save_dir+"_current_amp.png",dpi=200)
    # print("current AMP HIOU:")
    # amp_hiou = calculate_iou(gan_ca, ca, rg=rg)
    # print(amp_hiou)
    # accelerator.log({"amp_hiou": amp_hiou})

    
class LambdaLR():
    '''
    Controls the learning rate decay
    '''
    def __init__(self, n_epochs, offset, decay_start_epoch):
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

def weights_init_normal(m):
    '''
    Weight initialization
    '''
    classname = m.__class__.__name__
    dev  = 0.02
    if classname.find('Conv1d') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, dev)
    if classname.find('ConvTranspose1d') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, dev)
    elif classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, dev)
    elif classname.find('BatchNorm1d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, dev)
        torch.nn.init.constant_(m.bias.data, 0.0)