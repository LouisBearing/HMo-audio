import os
import argparse
import numpy as np
import json
import time
from metrics import *
from nets.utils import *
from nets.trainer import *
from dataset.vox_lips_dataset import *
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


config = Config()
parser = argparse.ArgumentParser()

### General params
parser.add_argument('--data_dir', default=r'vox/train', type=str)
parser.add_argument('--out_dir', type=str)
parser.add_argument('--lip_syncer', default=config.lip_syncer, type=str)
parser.add_argument('--lip_syncer_metrics', default=config.lip_syncer_metrics, type=str)
parser.add_argument('--lip_syncer_path', type=str, default='lip_syncer')

parser.add_argument('--keypoints', default=config.keypoints, type=bool_parser)

parser.add_argument('--data_dim', default=config.data_dim, type=int)
parser.add_argument('--obs_len', default=config.obs_len, type=int)
parser.add_argument('--smooth', default=config.smooth, type=bool_parser)

parser.add_argument('--fomm_confp', type=str, default=r'fomm_models/vox-adv-256.yaml')
parser.add_argument('--fomm_chkptp', type=str, default=r'fomm_models/vox-adv-cpk.pth.tar')

### Architecture params

# Embedders
parser.add_argument('--expansion_fact_do', default=config.expansion_fact_do, type=int)
parser.add_argument('--nblocks_do', default=config.nblocks_do, type=int)
parser.add_argument('--nblocks_frame_D', default=config.nblocks_frame_D, type=int)
parser.add_argument('--audio_dim', default=config.audio_dim, type=int)
parser.add_argument('--coord_dim', default=config.coord_dim, type=int)
parser.add_argument('--coord_dim_D', default=config.coord_dim_D, type=int)
parser.add_argument('--audio_fpn', default=config.audio_fpn, type=bool_parser)
parser.add_argument('--fpn_inner_dim', default=config.fpn_inner_dim, type=int)
parser.add_argument('--streams_merging_activation', default=config.streams_merging_activation, type=str)
parser.add_argument('--in_dim_coord_do', default=config.in_dim_coord_do, type=int)

# Embedding dimensions
parser.add_argument('--hidden_size', default=config.hidden_size, type=int)
parser.add_argument('--hidden_size_D', default=config.hidden_size_D, type=int)
parser.add_argument('--hidden_size_Dvis', default=config.hidden_size_Dvis, type=int)

# Decoder
parser.add_argument('--seq_len', default=config.seq_len, type=int)
parser.add_argument('--nlayers_tf', default=config.nlayers_tf, type=int)
parser.add_argument('--nheads_tf', default=config.nheads_tf, type=int)
parser.add_argument('--pyramid_layers_g', default=config.pyramid_layers_g, type=int)

### Losses

parser.add_argument('--adv_loss_weight', default=config.adv_loss_weight, type=float)
parser.add_argument('--sup_loss_weight', default=config.sup_loss_weight, type=float)
parser.add_argument('--reco_loss_weight', default=config.reco_loss_weight, type=float)
parser.add_argument('--rigid_loss_weight', default=config.rigid_loss_weight, type=float)
parser.add_argument('--rigid_loss_type', default=config.rigid_loss_type, type=str)
parser.add_argument('--lip_loss_weight', default=config.lip_loss_weight, type=float)
parser.add_argument('--seq_stream_weight', default=config.seq_stream_weight, type=float)
parser.add_argument('--frame_weight', default=config.frame_weight, type=float)
parser.add_argument('--syncer_pyramid', default=config.syncer_pyramid, type=bool_parser)
parser.add_argument('--pyramid_style', default=config.pyramid_style, type=str)
parser.add_argument('--syncer_pyramid_kernel', default=config.syncer_pyramid_kernel, type=int)
parser.add_argument('--loss_pyramid_kernel', default=config.loss_pyramid_kernel, type=int)
parser.add_argument('--vis_loss_weight', default=config.vis_loss_weight, type=float)
parser.add_argument('--vis_loss_step', default=config.vis_loss_step, type=int)
parser.add_argument('--first_order_loss_weight', default=config.first_order_loss_weight, type=float)

### Discriminator params

parser.add_argument('--dis_config', default=config.dis_config, type=int)
parser.add_argument('--dis_type', default=config.dis_type, type=int)
parser.add_argument('--nlayers_D', default=config.nlayers_D, type=int)
parser.add_argument('--nheads_D', default=config.nheads_D, type=int)

### Learning params

parser.add_argument('--n_epochs', default=config.n_epochs, type=int)
parser.add_argument('--batch_size', default=config.batch_size, type=int)

# Optimizers
parser.add_argument('--learning_rate_g', default=config.learning_rate_g, type=float)
parser.add_argument('--learning_rate_d', default=config.learning_rate_d, type=float)
parser.add_argument('--adam_beta_1', default=config.adam_beta_1, type=float)

# Schedulers
parser.add_argument('--lr_type', default=config.lr_type, type=str)
parser.add_argument('--gamma_lr', default=config.gamma_lr, type=float)
parser.add_argument('--step_iter_lr', default=config.step_iter_lr, type=int)
parser.add_argument('--gamma_lr_D', default=config.gamma_lr_D, type=float)
parser.add_argument('--step_iter_lr_D', default=config.step_iter_lr_D, type=int)



## Arguments parsing and config parameters setting
args = parser.parse_args()
init_distributed_mode(args)
for attr, attr_value in args.__dict__.items():
    setattr(config, attr, attr_value)

## Dataset instantiation
dataset = VoxLipsDataset(args.data_dir)
train_set, val_set = random_split(dataset, [int(len(dataset) * 0.98), len(dataset) - int(len(dataset) * 0.98)])

loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=4, collate_fn=collate_vox_lips, drop_last=True)
val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=True, num_workers=4, collate_fn=collate_vox_lips, drop_last=True)

## Model instanciation and resuming
trainer = AudioHMoTrainer(config).cuda()
metrics = Metrics(config.lip_syncer_path, mode='kp_syncnet')
# Util pool function
pool = nn.AdaptiveAvgPool2d(224).cuda()


## Looking for a previous checkpoint
epoch = 0
steps = 0
out_dir = os.path.join('models', config.out_dir)
if os.path.exists(os.path.join(out_dir, f'model_chkpt.pt')):
    loader, val_loader, epoch, steps = trainer.resume(out_dir)
else:
    os.mkdir(out_dir)
    # Save the configuration
    with open(os.path.join(out_dir, 'args'), 'w') as f:
        json.dump(args.__dict__, f)

## Tensorboard
writer = SummaryWriter(out_dir)

## Training
save_steps = [40000, 70000, config.step_iter_lr * 200 - 2000, 84000]
best_score = 0
dis_losses = ['loss_seq', 'loss_frame', 'all_f_out', 'all_r_out']
gen_losses = ['loss_seq', 'loss_frame', 'lip_loss', 'supervised_loss', 'first_order_loss']
losses = {
    'dis_losses': {key: 0 for key in dis_losses},
    'gen_losses': {key: 0 for key in gen_losses}
}

while epoch < config.n_epochs:
    
    for batch in loader:
        
        ## Dis update
        out = trainer.dis_update(batch)
        for key in losses['dis_losses'].keys():
            losses['dis_losses'][key] += out[key]
    
        ## Gen update
        out = trainer.gen_update(batch)
        for key in losses['gen_losses'].keys():
            losses['gen_losses'][key] += out[key]

        if (steps % 50 == 0):
            for key in losses['dis_losses'].keys():
                writer.add_scalar(f'DLoss/{key}', losses['dis_losses'][key] / 50, global_step=steps)
                losses['dis_losses'][key] = 0
            for key in losses['gen_losses'].keys():
                writer.add_scalar(f'GLoss/{key}', losses['gen_losses'][key] / 50, global_step=steps)
                losses['gen_losses'][key] = 0
    
        ## Validation
        step_condition = False
        save_condition = False
        # if args.distributed:
        if (steps % 200 == 0) and (steps > 0):
            step_condition = True
        if (steps % 1000 == 0) or (steps > 65000 and steps % 200 == 0) or (steps in save_steps):
            save_condition = True
        if step_condition:
            gen_lr, dis_lr = trainer.step_scheduler()
            writer.add_scalar('GenLr', gen_lr, global_step=steps)
        if save_condition:
            best = 'none'
            metrics_dict = {'lse-c': 0, 'lse-d': 0, 'lse-off': 0}
            for lvl in range(1, 3):
                metrics_dict.update({
                    f'lse-c_{str(lvl)}': 0, f'lse-d_{str(lvl)}': 0, f'lse-off_{str(lvl)}': 0
                })
            t = time.time()

            for i, val_batch in enumerate(val_loader):
                if i > 1:
                    break
                with torch.no_grad():
                    out = trainer.forward(val_batch, seq_len=80)
                metrics_out = metrics.forward_train(out)
                for key in metrics_dict.keys():
                    metrics_dict[key] += metrics_out[key]

            for key in metrics_dict.keys():
                metrics_dict[key] = np.round(metrics_dict[key] / i, 3)
                print(f'key: {key}, value: {metrics_dict[key]}')
                writer.add_scalar(f'Metrics/{key}', metrics_dict[key], global_step=steps)
            if (metrics_dict['lse-c'] < best_score) and (steps > config.step_iter_lr * 200):
                best_score = metrics_dict['lse-c']
                best = 'lse-c'
            trainer.save(out_dir, loader, val_loader, steps, epoch, new_file=steps in save_steps, best=best)
        
        steps += 1

    epoch += 1

trainer.save(out_dir, loader, val_loader, steps, epoch, new_file=True, best='none')