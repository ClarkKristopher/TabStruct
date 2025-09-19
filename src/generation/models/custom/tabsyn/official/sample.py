import argparse
import os
import time
import warnings

import numpy as np
import torch

from .diffusion_utils import sample
from .latent_utils import get_input_generate, recover_data, split_num_cat_target
from .model import MLPDiffusion, Model
from .vae.model import Decoder_model

warnings.filterwarnings("ignore")


def main(args):
    device = args.device
    save_path = args.save_path

    train_z, _, _, ckpt_path, info, num_inverse, cat_inverse = get_input_generate(args)
    in_dim = train_z.shape[1]

    mean = train_z.mean(0)

    denoise_fn = MLPDiffusion(in_dim, 1024).to(device)

    model = Model(denoise_fn=denoise_fn, hid_dim=train_z.shape[1]).to(device)

    model.load_state_dict(torch.load(f"{ckpt_path}/model.pt"))

    """
        Generating samples    
    """
    start_time = time.time()

    num_samples = train_z.shape[0]
    sample_dim = in_dim

    x_next = sample(model.denoise_fn_D, num_samples, sample_dim)
    x_next = x_next * 2 + mean.to(device)

    syn_data = x_next.float().cpu().numpy()
    syn_num, syn_cat, syn_target = split_num_cat_target(syn_data, info, num_inverse, cat_inverse, args.device)

    syn_df = recover_data(syn_num, syn_cat, syn_target, info)

    idx_name_mapping = info["idx_name_mapping"]
    idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}

    syn_df.rename(columns=idx_name_mapping, inplace=True)
    syn_df.to_csv(save_path, index=False)

    end_time = time.time()
    print("Time:", end_time - start_time)

    print("Saving sampled data to {}".format(save_path))


def sample_without_recover_data(
    log_dir,
    device,
    # VAE predecoder
    d_numerical,
    categories,
    num_layers: int = 2,
    d_token: int = 4,
    n_head: int = 1,
    factor: int = 32,
):
    # === Prepare hidden states by VAE ===
    embedding_save_path = os.path.join(log_dir, "ckpt", "vae", "train_z.npy")
    train_z = torch.tensor(np.load(embedding_save_path)).float()
    train_z = train_z[:, 1:, :]
    B, num_tokens, token_dim = train_z.size()
    in_dim = num_tokens * token_dim
    train_z = train_z.view(B, in_dim)
    mean = train_z.mean(0)

    # === Load trained model ===
    ckpt_path = os.path.join(log_dir, "ckpt", "tabsyn")
    denoise_fn = MLPDiffusion(in_dim, 1024).to(device)
    model = Model(denoise_fn=denoise_fn, hid_dim=train_z.shape[1]).to(device)
    model.load_state_dict(torch.load(f"{ckpt_path}/model.pt"))

    # === Generating samples ===
    start_time = time.time()

    num_samples = train_z.shape[0]
    sample_dim = in_dim

    x_next = sample(model.denoise_fn_D, num_samples, sample_dim)
    x_next = x_next * 2 + mean.to(device)

    syn_data = x_next.float().cpu().numpy()

    # === Transform the generated data to the input feature space ===
    # Note: This is not recover data, but just decoding the hidden states
    pre_decoder = Decoder_model(num_layers, d_numerical, categories, d_token, n_head=n_head, factor=factor).to(device)
    decoder_save_path = os.path.join(log_dir, "ckpt", "vae", "decoder.pt")
    pre_decoder.load_state_dict(torch.load(decoder_save_path))

    syn_data = syn_data.reshape(syn_data.shape[0], -1, token_dim)
    norm_input = pre_decoder(torch.tensor(syn_data).to(device))
    x_hat_num, x_hat_cat = norm_input

    syn_cat = []
    for pred in x_hat_cat:
        syn_cat.append(pred.argmax(dim=-1))

    syn_num = x_hat_num.cpu().detach().numpy()
    if len(syn_cat) == 0:
        syn_cat = np.zeros((syn_num.shape[0], 0))
    else:
        syn_cat = torch.stack(syn_cat).t().cpu().numpy()

    end_time = time.time()
    print("Time:", end_time - start_time)

    return {
        "syn_num": syn_num,
        "syn_cat": syn_cat,
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generation")

    parser.add_argument("--dataname", type=str, default="adult", help="Name of dataset.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index.")
    parser.add_argument("--epoch", type=int, default=None, help="Epoch.")
    parser.add_argument("--steps", type=int, default=None, help="Number of function evaluations.")

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f"cuda:{args.gpu}"
    else:
        args.device = "cpu"
