import torch
import torch.nn as nn
import torch.nn.functional as F
import accelerate
import argparse
from tools import select_quantile, load_data, num_paramters, get_ca, get_tail_slope, inf_train_gen, LambdaLR, weights_init_normal, compute_normalized_tail_slope, normalize_tail_slope, calculate_iou
from network import Diffusion_PUNet # PositionalUNet, 
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from IPython.display import HTML
from collections import OrderedDict
# from accelerate import set_seed

accelerator = accelerate.Accelerator()
m = Diffusion_PUNet()
m.eval()
m = accelerator.prepare(m)

single_GPU = True
state_dict = torch.load("pre_trained_diffusion.pth")

if single_GPU:
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] 
        new_state_dict[name] = v
    m.load_state_dict(new_state_dict)
else:
    m.load_state_dict(state_dict)

## this function is for after having a trained model
SEQ_LEN = 800
beta1 = 1e-4
beta2 = 0.02
timestep = 500
b_t = (beta2 - beta1) * torch.linspace(0, 1, timestep+1, device=accelerator.device) + beta1 ## beta1 -> beta2 in 500 time steps. torch.linspace works as percentage. 
a_t = 1 - b_t ## flipped b_t. (1-beta1) -> (1-beta2) which is decreasing. 
ab_t = torch.cumsum(a_t.log(), dim=0).exp() # a_t is decreasing => a_t.log() is decreasing fast => ab_t is getting close to 1
ab_t[0] = 1
eps = 1e-8

def denoise(diffused, pred_noise, t):
    # less_diffused = 1 / torch.sqrt(ab_t[t]) * (diffused - (1 - torch.sqrt(ab_t[t])) * pred_noise)
    # less_diffused = (diffused - (1 - ab_t.sqrt()[t]) * pred_noise) / ab_t.sqrt()[t]
    # print(torch.sqrt(ab_t[t] + eps) == torch.sqrt(ab_t[t] + eps))
    # less_diffused = 1 / torch.sqrt(a_t[t]) * (diffused - ((1 - a_t[t]) / (torch.sqrt(1 - ab_t[t]))) * pred_noise)
    # z = torch.randn_like(diffused)
    # noise = b_t.sqrt()[t] * z
    less_diffused = (diffused - pred_noise * ((1 - a_t[t]) / (1 - ab_t[t]).sqrt())) / a_t[t].sqrt() # official one? ## add noise? 
    return less_diffused

def sample_ddpm(n_sample, translate=None, save_rate=20):
    if translate is not None:
        samples = translate
    else:
        samples = torch.randn(n_sample, 1, SEQ_LEN, device=accelerator.device) ## how come the n_sample is here? 
    intermediate = []

    for t in range(timestep, 0, -1):
        t_list = torch.tensor([t] * n_sample, device=accelerator.device) ## Generating only one image at a time, could speed this up? 
        pred_noise = m(samples, t_list / timestep)
        samples = denoise(samples, pred_noise, t_list[:, None, None])
        if t % save_rate == 0:
            intermediate.append(samples.squeeze().detach().cpu())
    intermediate = np.stack(intermediate)
    return samples, intermediate

if __name__ == "__main__":
    train_loader, test_loader = load_data(4, MJD=False, validation_split=0.2)
    plt.close()
    train_loader = accelerator.prepare(train_loader)
    for real_A, real_B, _ in train_loader:
        # plt.plot(range(800), real_A[0].squeeze())
        # samples, inter = sample_ddpm(real_B.size(0), translate=real_B)
        samples, inter = sample_ddpm(real_A.size(0), translate=real_A)
        break
    # samples, inter = sample_ddpm(4)
    # print(samples)
    # plt.close()
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    for i, ax in enumerate(axs.flatten()):
        if i >= samples.size(0):
            break
        ax.plot(range(800), samples[i].squeeze().detach().cpu())
    plt.tight_layout()
    plt.savefig("generated_from_detector_data.png", dpi = 300)

## problem:
## training and testing works fine. But generating from only noise has trouble.

# def norm_all(store, n_t, n_s):
#     # runs unity norm on all timesteps of all samples
#     nstore = np.zeros_like(store)
#     for t in range(n_t):
#         for s in range(n_s):
#             nstore[t,s] = unorm(store[t,s])
#     return nstore

# def unorm(x):
#     # unity norm. results in range of [0,1]
#     # assume x (h,w,3)
#     xmax = x.max((0,1))
#     xmin = x.min((0,1))
#     return(x - xmin)/(xmax - xmin)

# def plot_sample(x_gen_store,n_sample,nrows,save_dir, fn,  w, save=False):
#     ncols = n_sample//nrows
#     sx_gen_store = np.moveaxis(x_gen_store,2,4)                               # change to Numpy image format (h,w,channels) vs (channels,h,w)
#     nsx_gen_store = norm_all(sx_gen_store, sx_gen_store.shape[0], n_sample)   # unity norm to put in range [0,1] for np.imshow
    
#     # create gif of images evolving over time, based on x_gen_store
#     fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True,figsize=(ncols,nrows))
#     def animate_diff(i, store):
#         print(f'gif animating frame {i} of {store.shape[0]}', end='\r')
#         plots = []
#         for row in range(nrows):
#             for col in range(ncols):
#                 axs[row, col].clear()
#                 axs[row, col].set_xticks([])
#                 axs[row, col].set_yticks([])
#                 plots.append(axs[row, col].imshow(store[i,(row*ncols)+col]))
#         return plots
#     ani = FuncAnimation(fig, animate_diff, fargs=[nsx_gen_store],  interval=200, blit=False, repeat=True, frames=nsx_gen_store.shape[0]) 
#     plt.close()
#     if save:
#         ani.save(save_dir + f"{fn}_w{w}.gif", dpi=100, writer=PillowWriter(fps=5))
#         print('saved gif at ' + save_dir + f"{fn}_w{w}.gif")
#     return ani

# # visualize samples
# plt.clf()
# samples, intermediate_ddpm = sample_ddpm(32)
# animation_ddpm = plot_sample(intermediate_ddpm,32,4,"./", "ani_run", None, save=True)
# HTML(animation_ddpm.to_jshtml())
