import numpy as np
import torch
import torch.nn as nn
from .resnet import FeaturePyramidNetwork, DeepMLP

deep_mlp_layers = {
    '4': [4],
    '6': [1, 1, 1, 1, 1, 1],
    '7': [1, 2, 2, 2], # 12M params
    '8': [2, 2, 2, 2], # 20M params
    '10': [1, 3, 3, 3], 
    '16': [3, 4, 6, 3] # 35M params
}

KEYPOINTS_DIM = 10
MAX_CONTEXT_LENGTH = 40

## Choose to run on GPU or CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class DynamicalModel(nn.Module):
    
    def __init__(self, config):
        super(DynamicalModel, self).__init__()
        
        self.config = config
        self.h_size = config.hidden_size

        ## Keypoint encoder
        self.input_dim = KEYPOINTS_DIM
        self.emb_dim = self.input_dim * config.data_dim

        self.encoder_x = nn.Sequential(*[
            nn.Linear(self.emb_dim, 272),
            nn.LayerNorm(272),
            nn.Linear(272, 4 * 272),
            nn.GELU(),
            nn.Linear(4 * 272, config.coord_dim)
        ])

        ## Audio encoder
        fpn_inner_dim = config.audio_dim
        self.encoder_a = FeaturePyramidNetwork(config.audio_dim, fpn_inner_dim, bottom_up_only=True)

        ## Deep output net / Multi-scale module
        do_in_dim = self.h_size + config.audio_dim + config.coord_dim
        self.instantiate_deep_output_net(do_in_dim)

        ## Temporal module
        tf_layer = nn.TransformerEncoderLayer(d_model=config.coord_dim, nhead=config.nheads_tf, dim_feedforward=1024, dropout=0)
        self.temp_module = nn.TransformerEncoder(tf_layer, num_layers=config.nlayers_tf)
        self.lin_out = nn.Linear(config.coord_dim, self.h_size)


    def instantiate_deep_output_net(self, in_dim):

        groups = self.config.pyramid_layers_g

        self.deep_output = DeepMLP(in_dim * groups, 272 * groups, layers=deep_mlp_layers[str(self.config.nblocks_do)][:-1], expansion_fact=self.config.expansion_fact_do, 
            groups=groups)
        self.mask = DeepMLP(272 * groups, self.input_dim * groups, layers=[deep_mlp_layers[str(self.config.nblocks_do)][-1]], expansion_fact=self.config.expansion_fact_do, 
            groups=groups)
        self.leaf = DeepMLP(272 * groups, self.input_dim * self.config.data_dim * groups, layers=[deep_mlp_layers[str(self.config.nblocks_do)][-1]], 
            expansion_fact=self.config.expansion_fact_do, groups=groups)

        # Bias term
        groups += 1
        self.deep_output_bias = DeepMLP(in_dim - self.config.audio_dim, 272, layers=deep_mlp_layers[str(self.config.nblocks_do)][:-1],
            expansion_fact=self.config.expansion_fact_do)
        self.mask_bias = DeepMLP(272, self.input_dim, layers=[deep_mlp_layers[str(self.config.nblocks_do)][-1]], expansion_fact=self.config.expansion_fact_do)
        self.leaf_bias = DeepMLP(272, self.input_dim * self.config.data_dim, layers=[deep_mlp_layers[str(self.config.nblocks_do)][-1]],
            expansion_fact=self.config.expansion_fact_do)

        # Final layer
        self.final_block = nn.Linear(self.input_dim * self.config.data_dim * groups, self.input_dim * self.config.data_dim)


    def embed_audio(self, audio):
        '''
        Audio pyramid input are processed by independent CNN encoders with increasing recep. field, then interpolated in the embedding space 
        back to a sequence of same duration as the higher res input --> ideally, audio inputs should get smoother as one moves upward in the pyramid
        '''

        tgt_len = int(audio.shape[1] / 4)
        fpn_out = self.encoder_a(audio)
        output = [
            torch.nn.functional.interpolate(tens, size=tgt_len, mode='linear', align_corners=True).transpose(1, 2) for tens in fpn_out
        ]
        return tuple(output)
    
    
    def deep_out_forward(self, deep_out_input):

        bckbone_out = self.deep_output(deep_out_input)

        leaf_output = self.leaf(bckbone_out)
        leaf_output = torch.stack(leaf_output.split(int(leaf_output.shape[-1] / self.config.pyramid_layers_g), dim=-1), dim=0)

        mask = (self.mask(bckbone_out).repeat_interleave(self.config.data_dim, dim=1))
        mask = torch.stack(mask.split(int(mask.shape[-1] / self.config.pyramid_layers_g), dim=-1), dim=0)

        temp_mod_out_dim = int(deep_out_input.shape[1] / self.config.pyramid_layers_g) - self.config.audio_dim
        bckbone_out_bias = self.deep_output_bias(deep_out_input[:, :temp_mod_out_dim])
        
        leaf_output_bias = self.leaf_bias(bckbone_out_bias)
        leaf_output = torch.cat([leaf_output_bias[None], leaf_output], dim=0)

        mask_bias = (self.mask_bias(bckbone_out_bias).repeat_interleave(self.config.data_dim, dim=1))
        mask = torch.cat([mask_bias[None], mask], dim=0)


        if self.config.streams_merging_activation == 'softmax':
            mask = torch.softmax(mask, dim=0)
        else:
            mask = torch.sigmoid(mask)
        out = mask * leaf_output

        return self.final_block(out.transpose(0, 1).flatten(start_dim=1))


    def reset_mask(self):
        setattr(self, 'soft_mask', [])
        

    def forward(self, inpt, audio):
        '''
        Params:
        ------
        'inpt' shape: bs, obs_len, 10 (keypoints dim), data_dim
        'audio' spectrogram of shape bs, 4 * (obs_len + seq_len), audio_dim

        Outputs:
        -------
        Reconstructed sequence, shape: bs, obs_len - 1, 10, data_dim
        Predicted sequence, shape: bs, seq_len, 10, data_dim
        '''

        bs, obs_len, _, _ = inpt.size()

        # Flattening of coordinate inputs
        running_inpt = inpt.flatten(start_dim=-2)
        
        # Encode audio -> bs, obs_len + seq_len, config.audio_dim
        seq_len = int(audio.shape[1] / 4) - obs_len
        audio_pyramid = self.embed_audio(audio)
        audio = audio_pyramid[0]

        #####
        ### Encoding of observed sequence / warm-up of the autoregressive model
        #####

        # Embedding 
        coord = self.encoder_x(running_inpt)
        # Temp module
        forward_mask = torch.triu(torch.ones(obs_len, obs_len), diagonal=1).bool().to(device)
        deep_out_input = self.lin_out(self.temp_module(coord.transpose(0, 1), mask=forward_mask).transpose(0, 1))
        # Mutli-scale module
        deep_out_input = torch.cat([deep_out_input, coord], dim=-1)
        deep_out_input = torch.cat([torch.cat([deep_out_input, a[:, 1:obs_len + 1]], dim=-1) for a in audio_pyramid], dim=-1)
        last_vel = self.deep_out_forward(deep_out_input.flatten(end_dim=1)).view(bs, obs_len, self.input_dim, self.config.data_dim)
        # Residual addition of instantaneous velocities
        last_pos = inpt + last_vel
        
        if last_pos.size(1) > 1:
            obs_outputs = last_pos[:, :-1]
        else:
            obs_outputs = torch.empty(0, 0, 0, 0).cuda()
        last_pos = last_pos[:, -1]

        #####
        ### Decoding of hidden sequence
        #####
        len_to_decode = seq_len - 1
        running_inpt = last_pos.view(bs, -1)
        outputs = [last_pos.unsqueeze(1)]

        for i in range(len_to_decode):
            
            coord_i = self.encoder_x(running_inpt)

            coord = torch.cat([coord, coord_i.unsqueeze(1)], dim=1)[:, -MAX_CONTEXT_LENGTH:]
            forward_mask = torch.triu(torch.ones(coord.shape[1], coord.shape[1]), diagonal=1).bool().cuda()
            h_n = self.lin_out(self.temp_module(coord.transpose(0, 1), mask=forward_mask).transpose(0, 1))
            h_n = h_n[:, -1]

            deep_out_input = torch.cat([h_n, coord_i], dim=-1)
            deep_out_input = torch.cat([torch.cat([deep_out_input, a[:, i + obs_len + 1]], dim=-1) for a in audio_pyramid], dim=-1)

            # Velocity output
            last_vel = self.deep_out_forward(deep_out_input).view(bs, self.input_dim, self.config.data_dim)
            last_pos = last_pos + last_vel

            running_inpt = last_pos.view(bs, -1)

            outputs.append(last_pos.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)

        return obs_outputs, outputs


########
### Discriminator networks

windows = {
    '1': [5, 10, 20, 40],
    '2': [3, 5, 8, 10, 13, 20, 26, 40]
}

class Discriminator(nn.Module):
    
    def __init__(self, config):
        super(Discriminator, self).__init__()
        
        self.config = config
        self.lambda_seq_stream = config.seq_stream_weight
        self.lambda_frame = config.frame_weight
        
        ## Coord encoder
        self.input_dim = KEYPOINTS_DIM
        self.emb_dim = self.input_dim * config.data_dim

        self.encoder_x = nn.Sequential(*[
            nn.Linear(self.emb_dim, 408),
            nn.LayerNorm(408),
            nn.Linear(408, 4 * 408),
            nn.GELU(),
            nn.Linear(4 * 408, config.coord_dim_D)
        ])

        # Individual landmarks realism score
        self.landmark_classifier = DeepMLP(config.coord_dim_D, 1, layers=deep_mlp_layers[str(self.config.nblocks_frame_D)], expansion_fact=4)
            
        # Window length for individual and joint streams
        self.windows = windows[str(config.dis_config)]

        # Sequential discriminators
        self.lstm = nn.LSTM(input_size=config.coord_dim_D, hidden_size=config.hidden_size_D)
        self.D_seq = ProjectionDisc(config.hidden_size_D)

        
    def forward(self, x_obs, x):

        #####
        ## Embedding
        #####
        bs, seq_len, _, _ = x.size()
        obs_len = x_obs.size(1)
        emb_x_inpt = torch.cat([x_obs, x], dim=1).flatten(start_dim=-2)

        # Embed x
        emb_x = self.encoder_x(emb_x_inpt.flatten(end_dim=1)).view(bs, obs_len + seq_len, -1)
        emb_x_obs, emb_x_gen = emb_x[:, :obs_len], emb_x[:, obs_len:]

        ####
        ### Frame discriminator
        ####
        frame_score = self.landmark_classifier(emb_x_gen.flatten(end_dim=1))

        ####
        ### Sequence discriminator
        ####
        output = self.forward_local_proj_dis(emb_x_obs, emb_x, obs_len, seq_len)
        
        return output, frame_score


    def forward_local_proj_dis(self, x_obs, x, obs_len, seq_len):
        ## Multi-scale window-based (projection) discriminator
        # y stands for the last conditioning hidden vector from which to compute dot product in projection dis  
        bs = len(x)

        # Computation of the conditioning vector
        _, (y, _) = self.lstm(x_obs.transpose(0, 1))
        y = y.squeeze()

        # Initialization of scores
        local_scores = torch.zeros(bs, 1).to(device)

        for idx, window in enumerate(self.windows):
            stride = int(window / 2)
            # Subsequences start indices
            local_idx = np.arange(max(-obs_len, stride - window), seq_len - window + 1, stride)
            overlapping_chunks = torch.cat([x[:, obs_len + i:obs_len + i + window, :] for i in local_idx], dim=0)
            _, (local_out, _) = self.lstm(overlapping_chunks.transpose(0, 1))
            local_scores += self.D_seq(local_out.squeeze(), y, distances=local_idx).mean(dim=0)

        return local_scores / (idx + 1)
    
    
    def compute_gen_loss(self, x, x_pred):

        x_obs = x[:, 1:self.config.obs_len, :, :self.config.data_dim] if not self.config.all_tf \
            else x[:, [0], :, :self.config.data_dim]

        f_seq, f_frame = self.forward(x_obs, x_pred)

        adv_loss_seq = -torch.mean(f_seq)
        adv_loss_frame = -torch.mean(f_frame)

        out = self.lambda_seq_stream * adv_loss_seq + self.lambda_frame * adv_loss_frame
        
        return (out, adv_loss_seq, adv_loss_frame)
        
        
    def compute_dis_loss(self, x, x_pred):

        x_gt = x[:, self.config.obs_len:, :, :self.config.data_dim]
        x_obs_gt = x[:, 1:self.config.obs_len, :, :self.config.data_dim] if not self.config.all_tf \
            else x[:, [0], :, :self.config.data_dim]
        x_obs = x_obs_gt

        f_seq, f_frame = self.forward(x_obs, x_pred)
        all_f_out = (f_seq.mean().item(), f_frame.mean().item())
        r_seq, r_frame = self.forward(x_obs_gt, x_gt)
        all_r_out = (r_seq.mean().item(), r_frame.mean().item())

        adv_loss_seq = torch.mean(torch.max(torch.zeros_like(f_seq), 1 + f_seq)) + \
                        torch.mean(torch.max(torch.zeros_like(r_seq), 1 - r_seq))
        adv_loss_frame = torch.mean(torch.max(torch.zeros_like(f_frame), 1 + f_frame)) + \
                        torch.mean(torch.max(torch.zeros_like(r_frame), 1 - r_frame))


        out = self.lambda_seq_stream * adv_loss_seq + self.lambda_frame * adv_loss_frame
        
        return (out, adv_loss_seq, adv_loss_frame, all_f_out[0], all_r_out[0])


class ProjectionDisc(nn.Module):
    
    def __init__(self, input_dim):
        super(ProjectionDisc, self).__init__()
        
        dimension = 2 * input_dim
        self.phi = LinearLayer(input_dim, dimension, 'relu')
        self.psi = LinearLayer(dimension, dimension, 'relu')
        self.V = LinearLayer(dimension, input_dim)
        self.A = LinearLayer(dimension, 1)
        self.beta = nn.Parameter(torch.ones(1).mul_(5), requires_grad=True)
        
    def forward(self, x, y, distances):
        # x dim: len(distances) * bs, dim
        # y dim: bs, dim
        batch_size = y.shape[0]

        distances = (torch.tensor(distances).clamp(1)).float().cuda()
        
        phi = self.phi(x)

        y = y.repeat(len(distances), 1)

        V_out = self.V(phi)
        
        projection = torch.matmul(y.unsqueeze(1), V_out.unsqueeze(2)).squeeze(1)
        coef = distances ** (-1 / self.beta)
        coef = coef.repeat_interleave(batch_size)
        
        return self.A(coef.unsqueeze(1) * projection + self.psi(phi)).view(len(distances), batch_size, 1)


class LinearLayer(nn.Module):
    
    def __init__(self, in_channels, out_channels, activation='none', norm='none', spectral_norm=True, dropout=0):
        super(LinearLayer, self).__init__()

        if spectral_norm:
            self.linear = nn.utils.spectral_norm(nn.Linear(in_channels, out_channels))
        else:
            self.linear = nn.Linear(in_channels, out_channels)
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'leaky':
            self.activation = nn.LeakyReLU()
        else:
            self.activation = None
            
        if norm == 'in':
            self.norm = nn.InstanceNorm1d(out_channels)
        elif norm == 'bn':
            self.norm = nn.BatchNorm1d(out_channels, momentum=0.5)
        elif norm == 'ln':
            self.norm = nn.LayerNorm(out_channels)
        else:
            self.norm = None
            
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
            
        
    def forward(self, x):
        x = self.linear(x)
        if self.norm is not None:
            if self.norm.__class__.__name__.find('InstanceNorm') > -1:
                x = self.norm(x.unsqueeze(1)).squeeze()
            else:
                x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x
