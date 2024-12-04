import os
import torch
import pickle
import numpy as np
from metrics import FIDMetrics, Metrics, get_ssim_psnr
from nets.utils import Config
import imageio
from argparse import ArgumentParser


def main(src_dirp, fvd_len=40):

    config = Config()
    setattr(config, 'i3d_path', './checkpoints/rgb_imagenet.pt')
    setattr(config, 'iv3_path', './checkpoints/pt_inception-2015-12-05-6726825d.pth')
    setattr(config, 'lip_syncer_path', './lip_syncer')
    metrics = Metrics(config, data_statistics_path='./data_statistics')

    # FVD
    activations = []
    s_dirs = os.listdir(src_dirp)
    for i, file in enumerate(s_dirs):
        s_dir = os.path.join(src_dirp, file)
        frames_p = os.listdir(s_dir) 
        if len(frames_p) < fvd_len:
            continue
        batch = []
        start_idx = 0
        for key in range(start_idx, start_idx + fvd_len):
            idx = format(key, '05d')  + ".png"

            f = os.path.join(s_dir, idx)
            laf = torch.Tensor(imageio.imread(f) / 255).cuda()
            batch.append(laf.permute(2, 0, 1))

        inpt_tens = torch.stack(batch, dim=1)[None] # Shape 1, 3, fvd_len, h, w
        with torch.no_grad():
            activations.append(metrics.i3d(inpt_tens).flatten(start_dim=1).cpu().numpy())
        if i % 50 == 0:
            print(f'i {i}, shape: {activations[-1].shape}')

    act = np.concatenate(activations)
    mean, cov = metrics.compute_statistics(act)

    with open(os.path.join('./data_statistics', f'fvd_visual_cov'), 'wb') as file:
        pickle.dump(cov, file)
    with open(os.path.join('./data_statistics', f'fvd_visual_mean'), 'wb') as file:
        pickle.dump(mean, file)


def compute_fid(src_dirp, name):
    metrics = FIDMetrics()
    fid = metrics.get_fid(src_dirp)
    print(f'******~~~~~ {name}, FID: {fid} ~~~~~*****')

def compute_fvd(src_dirp, name, fvd_len):
    metrics = FIDMetrics()
    fvd = metrics.get_fvd(src_dirp, fvd_len)
    print(f'******~~~~~ {name}, FVD: {fvd} ~~~~~*****')

def compute_ssim_psnr(pred_dirp, src_dirp, name, n_imgs):
    ssim, psnr = get_ssim_psnr(pred_dirp, src_dirp, n_imgs)
    print(f'******~~~~~ {name}, SSIM: {ssim}, PSNR: {psnr} ~~~~~*****')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--img_dir", help='Path dir with images')
    parser.add_argument("--img_dir2", help='Path dir with src images for SSIM / PSNR calculation')
    parser.add_argument("--fid", action='store_true')
    parser.add_argument("--fvd", action='store_true')
    parser.add_argument("--ssim", action='store_true')
    parser.add_argument("--name", type=str)
    parser.add_argument("--length", default=40, type=int)
    args = parser.parse_args()
    print('Arguments received, calling main !')
    if args.fid:
        compute_fid(args.img_dir, args.name)
    elif args.fvd:
        compute_fvd(args.img_dir, args.name, args.length)
    elif args.ssim:
        compute_ssim_psnr(args.img_dir, args.img_dir2, args.name, args.length)
    else:
        main(args.img_dir, args.length)