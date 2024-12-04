import numpy as np
import torch
import torch.nn as nn
import os
import pickle
from torch.utils.data import DataLoader
from .pytorch_i3d import *
from .inception import *
from .resnet import *
from lip_syncer_train import *
# from ..lip_syncer_train import *
from fomm.modules.keypoint_detector import KPDetector
from fomm.modules.generator import OcclusionAwareGenerator
from scipy.spatial.transform import Rotation as R
import torch.distributed as dist
import yaml

syncer_pyramids = {
    '1': [1, 2, 3],
    '2': [2, 3, 4],
    '3': [1, 2, 3, 4]
}


class Config:
    
    def __init__(self):
        
        ### General params
        self.keypoints = False

        # Data
        self.data_dim = 2
        self.obs_len = 3
        self.seq_len = 40
        self.smooth = False

        ### Gen params

        # Networks
        self.net_type = 'rnn'
        self.all_tf = False

        # Embedders
        self.encoder_type = 1
        self.expansion_fact_do = 4
        self.nblocks_do = 4
        self.nblocks_frame_D = 4
        self.audio_dim = 512
        self.coord_dim = 512
        self.coord_dim_D = 512
        self.syncer_pyramid_kernel = 7
        self.loss_pyramid_kernel = 7
        self.pyramid_layers_g = 4
        self.interp_mode = 'linear'
        self.audio_fpn = False
        self.fpn_inner_dim = 0
        self.streams_merging_activation = 'softmax'
        self.in_dim_coord_do = 256
        self.bias_stream = False
        self.final_block = False

        # LSTMs and Transfo FF
        self.hidden_size = 64
        
        # Others
        self.init_h0_mode = 'random'
        self.lip_syncer = 'info'
        self.syncer_training_style = ''
        self.lip_syncer_metrics = 'syncnet'
        self.inpt_coord_do = True
        self.rdm_seq_len = False

        # Transfo
        self.nlayers_tf = 1
        self.nheads_tf = 1

        ### Losses
        
        self.adv_loss_weight = 1.0
        self.sup_loss_weight = 0.01
        self.reco_loss_weight = 1e-3
        self.rigid_loss_weight = 1.0
        self.rigid_loss_type = 'init'
        self.lip_loss_weight = 1.0
        self.syncer_pyramid = False
        self.pyramid_style = '1'
        self.vis_loss_weight = 0.0
        self.vis_loss_step = 0
        self.first_order_loss_weight = 0

        # Adv term breakdown
        self.seq_stream_weight = 1.0
        self.frame_weight = 1.0


        ### Discriminator params

        self.net_type_D = 'rnn'
        self.dis_config = 1
        self.Dvis_config = 2
        self.hidden_size_D = 1024
        self.nlayers_D = 1
        self.nheads_D = 1
        self.dis_type = 1
        self.hidden_size_Dvis = 2048
        
        ### Learning params
        
        self.n_epochs = 600
        self.batch_size = 50
        
        # Optimizers
        self.learning_rate_g = 2e-5
        self.learning_rate_d = 1e-5
        self.adam_beta_1 = 0
        
        # Schedulers
        self.lr_type = 'step_lr'
        self.gamma_lr = 1.0
        self.step_iter_lr = 2000
        self.gamma_lr_D = 1.0
        self.step_iter_lr_D = 500

        self.data_parallel = False


# Utils

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.weight.data.normal_(0.0, 0.02)
    if (classname.find('Linear') != -1) & (classname.find('LinearLayer') == -1):
        nn.init.kaiming_normal_(m.weight)
    if (classname.find('Conv2d') != -1):
        nn.init.kaiming_normal_(m.weight)
    if (classname.find('Conv1d') != -1):
        nn.init.kaiming_normal_(m.weight)


def load_syncer_pyramid(model_path, mode='info', is_pyramid=False, layers=None, kernel_size=3):
    no_lips = 'nolips' in mode
    pyramid = {'0': load_syncer_model(model_path, mode, no_lips=no_lips)}
    if is_pyramid:
        for lvl in layers:
            pyramid.update({str(lvl): load_syncer_model(model_path, f'{mode}_lvl{str(lvl)}_k{str(kernel_size)}', no_lips=no_lips)})
    # lip_mode = mode if (mode == 'hdtf') else 'syncnet'
    # pyramid.update({'lips': load_syncer_model(model_path, f'{lip_mode}_lips', True)})
    return nn.ModuleDict(pyramid)


def load_syncer_model(model_path, mode='info', no_lips=False):
    add = f'_{mode}'
    with open(os.path.join(model_path, 'args'), 'rb') as f:
        args = pickle.load(f)
    parameters = {
        'resnet': dict(pretrained=False, pretrained_path='', num_classes=args.e_dim),
        'conv1d_x': dict(inplanes=272, block=InvResX1D, layers=layers[str(args.conv_layers)], out_dim=args.e_dim),
        'conv1d_a': dict(inplanes=a_dim[str(args.audio_style)], block=InvResX1D, layers=layers[str(args.conv_layers)], out_dim=args.e_dim),
        'conv2d': dict(inplanes=1, block=InvResX2D, layers=layers[str(args.conv_layers)], out_dim=args.e_dim)
    }
    setattr(args, 'only_lips', False)
    setattr(args, 'no_lips', no_lips)
    setattr(args, 'keypoints', 'kp' in mode)
    params = parameters[args.e_x], parameters[args.e_a]
    model = LipSyncNet(args, params)
    save_dict = torch.load(os.path.join(model_path, f'model_chkpt{add}.pt'))
    model.load_state_dict(save_dict['checkpoints'])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def moving_avg_torch(a, n):
    '''
    Moving average on axis 0
    '''
    if n == 0:
        return a
    b = torch.cumsum(a, dim=0)
    b[n:] = b[n:] - b[:-n]
    return torch.cat([a[:n - 1], b[n - 1:] / n])


def moving_avg_with_reflect_pad_torch(a, n):
    '''
    Moving average on axis 0
    '''
    if n == 0:
        return a
    n_pads = int((n - 1) / 2)
    padding = (n_pads, n_pads + 1)
    for i in range(len(a.shape) - 1):
        padding = (0,0) + padding
    b = nn.functional.pad(a[None], padding, mode='reflect')[0]
    b = torch.cumsum(b, dim=0)
    b[n:] = b[n:] - b[:-n]
    return b[n - 1:n - 1 + len(a)] / n


def bool_parser(string):
    if string.lower() == 'false':
        return False
    return True


## Rotation & translation functions for batched Tensors

def get_R_matrix_from_tensor(tens):
    '''
    Returns rotation matrix that frontalizes first face image in a sequence
    params:
    ------
    tens: Tensor of shape bs, seq_len, 68, 3
    '''
    # Project on y axis and normalize
    tens = tens.cpu()
    proj_y = (tens[:, 0, 0] - tens[:, 0, 16])[:, [0, 2]]
    sign = (proj_y[:, 0] / torch.abs(proj_y[:, 0])).unsqueeze(-1)
    proj_y = sign * proj_y / torch.norm(proj_y, dim=-1).unsqueeze(1)
    
    sin_half = (proj_y[:, 1] / torch.abs(proj_y[:, 1])) * ((0.5 * (1 - proj_y[:, 0])) ** 0.5)
    cos_half = (0.5 * (1 + proj_y[:, 0])) ** 0.5
    ry = torch.Tensor(np.array([R.from_quat([0, sin_half[i], 0, cos_half[i]]).as_matrix() for i in range(len(sin_half))])).cuda()
    return ry

def T_matrix_from_tensor(origin):
    '''
    origin of shape bs, 3
    '''
    return torch.cat([torch.eye(3).repeat(origin.shape[0], 1, 1).cuda(), origin.unsqueeze(-1)], dim=-1)

def translate_tensor(vector, M):
    ones = torch.ones_like(vector)[..., [0]].cuda()
    return torch.bmm(torch.cat([vector, ones], dim=-1).flatten(start_dim=1, end_dim=2),
          M.transpose(1, 2)).view(vector.shape)

def b_rotate_3D_tensor(tensor, sin_half=None, cos_half=None):
    '''
    tens: Tensor of shape bs, seq_len, 68 * 3 (dim) * 3 (x, v, a)
    '''

    # Center position tensor
    bs, seq_len, input_dim = tensor.shape
    positions = tensor[..., :68 * 3].contiguous().view(bs, seq_len, 68, 3)
    origin = positions.mean(dim=(1, 2))
    centerred_tensor = translate_tensor(positions, T_matrix_from_tensor(-origin))

    # Construct rotation matrix bs * 3 * 3
    if sin_half is None:
        with torch.no_grad():
            rotation_matrix = get_R_matrix_from_tensor(centerred_tensor)
    else:
        rotation_matrix = torch.Tensor(np.array([R.from_quat([0, sin_half[i], 0, cos_half[i]]).as_matrix() for i in range(len(sin_half))])).cuda()

    rotated_centerred_tensor = torch.bmm(centerred_tensor.flatten(start_dim=1, end_dim=2),
              rotation_matrix.transpose(1, 2)).view(centerred_tensor.shape)
    rotated_tensor = translate_tensor(rotated_centerred_tensor, T_matrix_from_tensor(origin))

    out = [rotated_tensor.flatten(start_dim=-2)]

    # Apply rotation on velocity and acceleration
    for i in range(1, tensor.shape[-1] // (3 * 68)):
        split = tensor[..., i * 3 * 68: (i + 1) * 3 * 68].contiguous().view(bs, seq_len, 68, 3)
        out.append(torch.bmm(split.flatten(start_dim=1, end_dim=2), rotation_matrix.transpose(1, 2)).view(bs, seq_len, -1))

    return torch.cat(out, dim=-1)

###

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    # args.distributed = False
    # return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, device: {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

def is_main_process():
    return dist.get_rank() == 0

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def load_KPDect(config_path, checkpoint_path):

    with open(config_path) as f:
        config = yaml.safe_load(f)

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    kp_detector.cuda()
    
    checkpoint = torch.load(checkpoint_path)
 
    kp_detector.load_state_dict(checkpoint['kp_detector'])
    kp_detector.eval()
    for param in kp_detector.parameters():
        param.requires_grad_(False)
    
    return kp_detector


def load_Gen(config_path, checkpoint_path):

    with open(config_path) as f:
        config = yaml.safe_load(f)

    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    generator.cuda()

    checkpoint = torch.load(checkpoint_path)
 
    generator.load_state_dict(checkpoint['generator'])
    generator.eval()
    for param in generator.parameters():
        param.requires_grad_(False)

    return generator


def encode_position(x):

    _, length, dim = x.shape
    device = x.device

    encoding = torch.matmul(torch.arange(1, 1 + length, dtype=torch.float32).unsqueeze(1),
                1 / (1e4 ** (torch.arange(0, dim, step=2) / dim)).unsqueeze(0))

    sin = torch.sin(encoding)
    cos = torch.cos(encoding)
    encoding = torch.stack([sin, cos], dim=-1).flatten(start_dim=-2)

    return encoding.to(device)