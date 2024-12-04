import os
import yaml
import torch
import glob
import imageio
import numpy as np
import json
import argparse
from skimage import img_as_ubyte
from nets.networks import DynamicalModel
from nets.utils import Config
from fomm.modules.keypoint_detector import KPDetector
from fomm.modules.generator import OcclusionAwareGenerator
from dataset.preprocess_audio import get_audio_feature_from_audio

## *** Enter fomm model paths here ***
chkpt_p = 'fomm_models/vox-adv-cpk.pth.tar'
conf_p = 'fomm_models/vox-adv-256.yaml'

conf_p =  r'/mnt/c/Users/laeri/FaceGeneration/first-order-model/config/vox-adv-256.yaml'
chkpt_p = r'/mnt/c/Users/laeri/FaceGeneration/first-order-model/models/vox-adv-cpk.pth.tar'

## Static STFT parameters
n_pts_coord = 5
n_pts_melspec = 20

## Choose to run on GPU or CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_KPDect(config_path, checkpoint_path):

    with open(config_path) as f:
        config = yaml.safe_load(f)

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    kp_detector.to(device)
    
    checkpoint = torch.load(checkpoint_path)
 
    kp_detector.load_state_dict(checkpoint['kp_detector'])
    kp_detector.eval()
    
    return kp_detector


def load_Gen(config_path, checkpoint_path):

    with open(config_path) as f:
        config = yaml.safe_load(f)

    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    generator.to(device)

    checkpoint = torch.load(checkpoint_path)
 
    generator.load_state_dict(checkpoint['generator'])
    generator.eval()
    
    return generator


def make_animation(source_image, driving_kp_sec, generator, kp_detector, out_p, save_n_imgs, max_l, img_path):

    with torch.no_grad():
        predictions = []
        ldk = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        source = source.to(device)
        kp_source = kp_detector(source)

        for frame_idx in range(len(driving_kp_sec)):
            if (max_l > 0) and (frame_idx == max(save_n_imgs, max_l)):
                break
            kp_driving = driving_kp_sec[frame_idx]
            out = generator(source, kp_source=kp_source, kp_driving=kp_driving)
            frame = np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]
            if (frame_idx < max_l) or (max_l == 0):
                predictions.append(frame)
            if frame_idx < save_n_imgs:
                imageio.imwrite(os.path.join(img_path, format(frame_idx, '05d')  + ".png"), (frame * 255).astype(np.uint8))
            
    imageio.mimsave(out_p, [img_as_ubyte(frame) for frame in predictions], fps=25) # needs imageio==2.22.0, imageio-ffmpeg==0.4.7
    

def add_audio(out_p, audio_p):
    vid_p = out_p.replace('_temp', '')
    cmd = [
        'ffmpeg',
        '-i', out_p,
        '-i', audio_p,
        '-t', '4.8', '-y',
        vid_p
    ]
    cmd = ' '.join(cmd)
    os.system(cmd)
    os.remove(out_p)


def load_model(model_p):

    config = Config()
    with open(os.path.join(model_p, 'args'), 'r') as f:
        args = json.load(f)
    for attr, attr_value in args.items():
        setattr(config, attr, attr_value)
    model = DynamicalModel(config).to(device)

    save_dict = torch.load(os.path.join(model_p, 'model_chkpt.pt'))
    model.load_state_dict(save_dict['checkpoints'])
    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)
    return model, config


def execute_model(model, config, kp0, melspec, seq_len):

    inc = 3.9943
    required_len = int(len(melspec) / inc)
    if seq_len > 0:
        required_len = min(1 + seq_len, required_len)

    i_x = np.arange(required_len)

    # Full spectrogram
    full_audio = melspec[:np.round((i_x[-1] + 1) * inc).astype(int)]

    # A dummy initial history is provided to warm up the autoregressive model, those time steps are later discarded
    prepad_l = np.round(4 * (config.obs_len - 1)).astype(int)
    mirror_pad = full_audio[:prepad_l, :].flip([0])

    full_audio = torch.nn.functional.pad(full_audio[None], (0, 0, 0, 5), mode='reflect')[:, :4 * required_len]
    mirror_pad = mirror_pad[None]

    inpt = kp0[:, None, :, :config.data_dim].repeat(1, config.obs_len, 1, 1)
    with torch.no_grad():
        x_rec, x_pred = model(inpt, torch.cat([mirror_pad, full_audio], dim=1))
    out_kp = torch.cat([x_rec[:, [-1]], x_pred], dim=1)

    return out_kp


def main(args):

    ## Step 1, load dyna model & fomm / fa models
    model, config = load_model(args.model_dir)
    kp_detector = load_KPDect(conf_p, chkpt_p)
    generator = load_Gen(conf_p, chkpt_p)
    print(f'Models successfully loaded')
    
    os.makedirs(args.out_dir, exist_ok=True)

    # Wav file required
    audio_ps = glob.glob(args.audio_dir + '/*.wav')

    print(f'{len(audio_ps)} files found, iterating....')

    ## Step 2, iterate through dataset, generate keypoints and reenact videos
    for audio_p in audio_ps:

        # Calculate melspec filter bank energy
        _, melspec = get_audio_feature_from_audio(args.audio_dir, os.path.basename(audio_p))
        melspec = torch.Tensor(melspec).to(device)
        
        # Search for initial frame at png format corresponding to the audio file
        img_id = os.path.basename(audio_p).replace('.wav', '')
        img0_p = os.path.join(args.img_dir, img_id + '.png')
        if not os.path.isfile(img0_p):
            print(f'File {img0_p} not found, going to next file...')
            continue

        # Extract keypoints
        img0 = imageio.imread(img0_p) / 255
        # img0 = imageio.v2.imread(img0_p) / 255
        source = torch.tensor(img0[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2).to(device)
        kp_source = kp_detector(source)
        kp0 = torch.cat([kp_source['value'], kp_source['jacobian'].flatten(start_dim=-2)], dim=-1)

        # Generate and save kp sequence
        if args.max_l > 0:
            seq_len = max(args.max_l, args.save_n_imgs)
        else:
            seq_len = 0
        kp_driving = execute_model(model, config, kp0, melspec, seq_len)

        if args.save_n_imgs > 0:
            img_path = os.path.join(args.out_dir, img_id.replace('#', '___'))
            os.makedirs(img_path, exist_ok=True)
        else:
            img_path = None
        
        # Reenact
        if len(kp_driving.shape) == 4:
            kp_driving = kp_driving.squeeze(0)
        values = kp_driving[..., :2]
        jacobian = kp_driving[..., 2:].reshape(len(kp_driving), 10, 2, 2)
        driving_kp_sec = [{'value': values[[i]], 'jacobian': jacobian[[i]]} for i in range(len(kp_driving))]
        
        out_p = os.path.join(args.out_dir, img_id + '_temp.mp4').replace('#', '__')
        make_animation(img0, driving_kp_sec, generator, kp_detector, out_p, args.save_n_imgs, args.max_l, img_path)
        # Add audio
        add_audio(out_p, audio_p)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_dir', help='Path to vox test ds')
    parser.add_argument('--img_dir', help='Path to first frames')
    parser.add_argument('--model_dir', help='Path to model weights & args file')
    parser.add_argument('--out_dir', help='Where to save output scores')
    parser.add_argument('--save_n_imgs', default=0, type=int, help='How many imgs to save')
    parser.add_argument('--max_l', default=0, type=int, help='Max number of frames')
    args = parser.parse_args()
    main(args)