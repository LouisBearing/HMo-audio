from .networks import *
from .utils import *
import numpy as np
import os
import torch.nn as nn
import pickle


class AudioHMoTrainer(nn.Module):
    
    def __init__(self, config, load_syncer=True):
        super(AudioHMoTrainer, self).__init__()
        
        self.config = config
        
        ## Dynamical Model
        self.dynamical_model = DynamicalModel(config)
        setattr(self, 'gradient_clipping_value', 0)

        ## Discriminator
        self.dis = Discriminator(config)

        ## LipSyncer
        if load_syncer:
            self.lip_syncer = load_syncer_pyramid(config.lip_syncer_path, mode='kpsync', is_pyramid=config.syncer_pyramid,
                layers=syncer_pyramids[config.pyramid_style], kernel_size=config.syncer_pyramid_kernel)
        
        ## Optimizers
        betas = (config.adam_beta_1, 0.999)
        self.optim_D_params = list(self.dis.parameters())
        self.optim_G_params = list(self.dynamical_model.parameters())
        self.optim_D = torch.optim.Adam(params=self.optim_D_params, betas=betas, lr=config.learning_rate_d)
        self.optim_G = torch.optim.Adam(params=self.optim_G_params, betas=betas, lr=config.learning_rate_g)

        ## Schedulers
        if config.lr_type == 'step_lr':
            self.gen_scheduler = torch.optim.lr_scheduler.StepLR(self.optim_G, config.step_iter_lr, config.gamma_lr)
        elif config.lr_type == 'exp_lr':
            self.gen_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optim_G, config.gamma_lr)
        self.dis_scheduler = torch.optim.lr_scheduler.StepLR(self.optim_D, config.step_iter_lr_D, config.gamma_lr_D)


    def forward(self, batch, seq_len=None):

        x, mid_aligned_audio, start_aligned_audio, full_audio, _ = self.prepare_train_batch(batch, seq_len)
        inpt = x[:, :self.config.obs_len, :, :self.config.data_dim]
        x_rec, x_pred = self.dynamical_model(inpt, full_audio)

        return x, x_rec, x_pred, start_aligned_audio, mid_aligned_audio, full_audio # x -> 0...obs_len + seq_len - 1, x_rec -> 1...obs_len - 1,
        # x_pred -> obs_len...obs_len + seq_len - 1


    def prepare_train_batch(self, batch, seq_len=None, test=False):

        ldk, mel, lengths = batch
        ldk, mel = ldk.cuda(), mel.cuda()

        if test:
            obs_len = 1
        else:
            obs_len = self.config.obs_len
        
        if seq_len is None:
            if self.config.rdm_seq_len:
                min_batch_len = min([lx for (lx, _) in lengths])
                seq_len = min(120, np.random.randint(min_batch_len, min_batch_len + 30))
                required_len = obs_len + seq_len
            else:
                required_len = obs_len + self.config.seq_len
        else:
            required_len = obs_len + seq_len

        n_lays = 1 if self.config.audio_fpn else 4

        splits = []
        idx_x, idx_a = [np.cumsum([0] + list(l)) for l in zip(*lengths)]
        increments = [l_a / l_x for l_x, l_a in lengths]

        prep_batch = []
        mid_aligned_audio = []
        full_audio = []
        init_audio_for_inference = []

        kept_id = []
        for k, (i_x0, i_xf, i_a0, i_af, inc) in enumerate(zip(idx_x[:-1], idx_x[1:], idx_a[:-1], idx_a[1:], increments)):
            if i_xf - i_x0 < required_len + 3:
                continue
            kept_id.append(k)
            m = mel[i_a0:i_af]
            x = ldk[i_x0:i_xf]

            # Sample subsequence of required length
            off = np.random.randint(0, len(x) - required_len)
            i_x = np.arange(off, off + required_len)
            inpt = x[i_x]
            
            # Full spectrogram
            a_seq = m[np.round(i_x[0] * inc).astype(int):np.round((i_x[-1] + 1) * inc).astype(int)]
            full_audio.append(a_seq)
            if test:
                prepad_l = np.round(inc * (self.config.obs_len - 1)).astype(int)
                mirror_pad = a_seq[:prepad_l, :].flip([0])
                a_seq = torch.cat([mirror_pad, a_seq], dim=0)
                init_audio_for_inference.append(mirror_pad)

            # Order melspec as a stacked sequence of spectro chunks time-aligned with each x inpt frame...
            npads = np.round(inc * (n_pts_coord - 1) / 2).astype(int)
            padded_a = nn.functional.pad(a_seq, (0, 0, npads, npads + 4), value=np.log(1e-8))
            i_a = np.round(np.arange(len(inpt)) * inc).astype(int)
            start_aligned_audio = torch.stack([padded_a[npads + i:npads + i + n_pts_melspec] for i in i_a[:-n_pts_coord + 1]])
            # ... and a pyramid of spectro chunks centered on each coordinate frame...
            audio_pyramid = tuple()
            for lvl in range(n_lays):
                i_a = np.round(np.arange(len(inpt) / (2 ** lvl)) * inc).astype(int)
                audio_pyramid += torch.stack([padded_a[i:i + n_pts_melspec] for i in i_a]),
                a_seq = moving_avg_with_reflect_pad_torch(a_seq, n=self.config.syncer_pyramid_kernel)[::2]
                padded_a = nn.functional.pad(a_seq, (0, 0, npads, npads + 4), value=np.log(1e-8))
            
            prep_batch.append((inpt, start_aligned_audio))
            mid_aligned_audio.append(audio_pyramid)
            
        inpt, start_aligned_audio = [torch.stack(_) for _ in zip(*prep_batch)]
        mid_aligned_audio = [torch.stack(_) for _ in zip(*mid_aligned_audio)]

        full_audio = torch.stack([
            torch.nn.functional.pad(tens[None], (0, 0, 0, 5), mode='reflect').squeeze()[:4 * required_len] for tens in full_audio
        ])
        if test:
            init_audio_for_inference = torch.stack(init_audio_for_inference)
        # An even batch size is required
        rounded_bs = 2 * int(0.5 * len(inpt))

        return inpt[:rounded_bs], tuple([tens[:rounded_bs] for tens in mid_aligned_audio]), start_aligned_audio[:rounded_bs], full_audio[:rounded_bs], \
            init_audio_for_inference


    def prepare_test_batch(self, batch, seq_len, filenames, sep_model_without_pyG=False):
        '''
        Generation starts from frame #0
        filenames is used to keep track of possibly discarded samples of unsufficient length
        '''

        ldk, mel, lengths = batch
        ldk, mel = ldk.cuda(), mel.cuda()

        # If no seqlen then it's a single sample
        if seq_len is None:
            seq_len = len(ldk) - 1

        required_len = 1 + seq_len

        n_lays = 1 if self.config.audio_fpn else 4

        splits = []
        idx_x, idx_a = [np.cumsum([0] + list(l)) for l in zip(*lengths)]
        increments = [l_a / l_x for l_x, l_a in lengths]

        prep_batch = []
        mid_aligned_audio = []
        full_audio = []
        init_audio_for_inference = []
        out_filename_list = []

        tgt_len = required_len
        if sep_model_without_pyG:
            tgt_len += self.config.obs_len - 1

        for i_x0, i_xf, i_a0, i_af, inc, fname in zip(idx_x[:-1], idx_x[1:], idx_a[:-1], idx_a[1:], increments, filenames):
            if i_xf - i_x0 < required_len: # + self.config.obs_len - 1
                continue
            out_filename_list.append(fname)
            m = mel[i_a0:i_af]
            x = ldk[i_x0:i_xf]
            i_x = np.arange(required_len)
            inpt = x[i_x]
            
            # Full spectrogram
            a_seq = m[:np.round((i_x[-1] + 1) * inc).astype(int)]
            full_audio.append(a_seq)
            prepad_l = np.round(4 * (self.config.obs_len - 1)).astype(int)
            mirror_pad = a_seq[:prepad_l, :].flip([0])
            if sep_model_without_pyG:
                a_seq = torch.cat([mirror_pad, a_seq], dim=0)
            init_audio_for_inference.append(mirror_pad)

            # Order melspec as a stacked sequence of spectro chunks time-aligned with each x inpt frame...
            npads = np.round(inc * (n_pts_coord - 1) / 2).astype(int)
            padded_a = nn.functional.pad(a_seq, (0, 0, npads, npads + 4), value=np.log(1e-8))
            i_a = np.round(np.arange(tgt_len) * inc).astype(int)
            start_aligned_audio = torch.stack([padded_a[npads + i:npads + i + n_pts_melspec] for i in i_a[:-n_pts_coord + 1]])
            # ... and a pyramid of spectro chunks centered on each coordinate frame...
            audio_pyramid = tuple()
            for lvl in range(n_lays):
                i_a = np.round(np.arange(tgt_len / (2 ** lvl)) * inc).astype(int)
                audio_pyramid += torch.stack([padded_a[i:i + n_pts_melspec] for i in i_a]),
                a_seq = moving_avg_with_reflect_pad_torch(a_seq, n=self.config.syncer_pyramid_kernel)[::2]
                padded_a = nn.functional.pad(a_seq, (0, 0, npads, npads + 4), value=np.log(1e-8))
            
            prep_batch.append((inpt, start_aligned_audio))
            mid_aligned_audio.append(audio_pyramid)
        
        if len(prep_batch) == 0:
            return None, None, None, None, None, None 

        inpt, start_aligned_audio = [torch.stack(_) for _ in zip(*prep_batch)]
        mid_aligned_audio = [torch.stack(_) for _ in zip(*mid_aligned_audio)]

        full_audio = torch.stack([
            torch.nn.functional.pad(tens[None], (0, 0, 0, 5), mode='reflect').squeeze()[:4 * required_len] for tens in full_audio
        ])
        init_audio_for_inference = torch.stack(init_audio_for_inference)

        return inpt, tuple([tens for tens in mid_aligned_audio]), start_aligned_audio, full_audio, init_audio_for_inference, out_filename_list


    def dis_update(self, batch):

        if self.config.adv_loss_weight == 0:
            out = {key: torch.tensor(0.0).cuda() for key in ['loss_seq', 'loss_frame', 'all_f_out', 'all_r_out']}
            adv_loss = torch.tensor(0.0).cuda()
            if self.config.vis_loss_weight == 0:
                return out

        self.optim_D.zero_grad()

        with torch.no_grad(): 
            x, x_rec, x_pred, _, audio, _ = self.forward(batch)
        
        (adv_loss, loss_seq, loss_frame, all_f_out, all_r_out) = self.dis.compute_dis_loss(x, x_pred)
        out = {'loss_seq': loss_seq.item(), 'loss_frame': loss_frame.item(),
            'all_f_out': all_f_out, 'all_r_out': all_r_out}

        adv_loss.backward()
        if self.gradient_clipping_value > 0:
            nn.utils.clip_grad_norm_(self.optim_D_params, self.gradient_clipping_value)
        self.optim_D.step()

        return out

    
    def gen_update(self, batch):

        self.optim_G.zero_grad()

        x, x_rec, x_pred, audio, mid_aligned_audio, full_audio = self.forward(batch)
        x_obs, x_gt = x[:, 1:self.config.obs_len, :, :self.config.data_dim], x[:, self.config.obs_len:, :, :self.config.data_dim]
        if len(x_rec) == 0:
            x_rec = x_obs

        out = {}
        # Adversarial loss
        loss_visual = torch.tensor(0.0).cuda()
        if self.config.adv_loss_weight == 0:
            adv_loss = torch.tensor(0.0).cuda()
            loss_seq = torch.tensor(0.0).cuda()
            loss_frame = torch.tensor(0.0).cuda()
        else:
            (adv_loss, loss_seq, loss_frame) = self.dis.compute_gen_loss(x, x_pred)

        loss = self.config.adv_loss_weight * adv_loss
        out.update({'loss_seq': loss_seq.item(), 'loss_frame': loss_frame.item()})

        # Lip Sync Loss
        x_0 = x[:, [0], :, :self.config.data_dim]
        x_0 = torch.cat([x_0, x_rec], dim=1)
        x_lips = torch.cat([x_0, x_pred], dim=1)[..., :self.config.data_dim if self.config.keypoints else 2]
        stacked_x_lips = torch.stack([x_lips[:, i:i + n_pts_coord] for i in range(x_lips.shape[1] - n_pts_coord + 1)], dim=1)
        lip_loss = self.lip_sync_loss(stacked_x_lips, audio, self.lip_syncer['0'])
        lip_p_loss, n_avg = self.lip_sync_pyramid_loss(x_lips, full_audio)
        lip_loss = (lip_loss + lip_p_loss) / (1 + n_avg)
        loss += self.config.lip_loss_weight * lip_loss
        out['lip_loss'] = lip_loss.item()

        # Supervised loss
        supervised_loss = self.l2_loss(x_pred, x_gt)
        if x_obs.shape[1] > 0:
            reconstruction_loss = self.l2_loss(x_rec, x_obs)
        else:
            reconstruction_loss = torch.tensor(0.0).cuda()
        loss += self.config.sup_loss_weight * supervised_loss + self.config.reco_loss_weight * reconstruction_loss
        out['supervised_loss'] = supervised_loss.item()

        # First order regularization
        first_order_loss = self.velocity_regularization_loss(x_pred)
        loss += self.config.first_order_loss_weight * first_order_loss
        out['first_order_loss'] = first_order_loss.item()
        
        # Rigid Loss
        if self.config.data_dim == 3:
            rigid_loss = self.rigid_loss(x_pred, x_obs)
        else:
            rigid_loss = torch.tensor(0.0).cuda()
        loss += self.config.rigid_loss_weight * rigid_loss
        out['rigid_loss'] = rigid_loss.item()

        loss.backward()

        if self.gradient_clipping_value > 0:
            nn.utils.clip_grad_norm_(self.optim_G_params, self.gradient_clipping_value)
        self.optim_G.step()

        ## Compute mouth opening = distance between landmarks 62 and 66 and compare with gt
        if not self.config.keypoints:
            mouth_op_pred = ((x_pred[..., 62, :] - x_pred[..., 66, :]).pow(2)).sum(dim=-1).pow(0.5)
            mouth_op_gt = ((x_gt[..., 62, :] - x_gt[..., 66, :]).pow(2)).sum(dim=-1).pow(0.5)
            out['mouth_dist'] = (mouth_op_pred - mouth_op_gt).abs().mean()
        else:
            out['mouth_dist'] = torch.tensor(0.0).cuda()

        return out


    def l2_loss(self, x, y, reduce=True):
        if reduce:
            return torch.mean(torch.sum((x - y)**2, dim=-1))
        else:
            return torch.mean(torch.sum((x - y)**2, dim=-1), dim=-1)


    def velocity_regularization_loss(self, x):
        """
        Expected x shape: bs, sl, npts, dim
        """
        return torch.abs(torch.diff(x, dim=1)).mean()


    def rigid_loss(self, x_pred, x_obs):

        fix_pts_index = np.array([
            0, 1, 2, 14, 15, 16, # skull
            27, 30, 31, 35, # nose
            36, 39, 42, 45 # eyes
        ])

        distances = x_pred[..., fix_pts_index, :]
        distances = ((distances.unsqueeze(3) - distances.unsqueeze(2)) ** 2).sum(dim=-1).clamp(min=1e-12) ** 0.5

        if self.config.rigid_loss_type == 'init':
            d_0 = x_obs[:, 0, fix_pts_index]
            d_0 = ((d_0.unsqueeze(2) - d_0.unsqueeze(1)) ** 2).sum(dim=-1).clamp(min=1e-12) ** 0.5
            d_0 = d_0.unsqueeze(1)
            loss = (torch.abs(distances - d_0) / d_0.clamp(min=1e-12)).mean()
        elif self.config.rigid_loss_type == 'delta':
            loss = (torch.abs(distances[:, 1:] - distances[:, :-1]) / distances[:, :-1].clamp(min=1e-12)).mean()

        return loss


    def lip_sync_pyramid_loss(self, x, a):
        
        loss = 0
        n_avg = 0
        max_level = max(syncer_pyramids[self.config.pyramid_style])
        keys = self.lip_syncer.keys()
        for lvl in range(1, max_level + 1):

            stacked_list = []
            a_new = []

            for idx in range(len(x)):
                s_x = x[idx]
                s_x = moving_avg_with_reflect_pad_torch(s_x, n=self.config.loss_pyramid_kernel)[::2] # possibility to smooth differently from the expert
                s_a = a[idx]
                s_a = moving_avg_with_reflect_pad_torch(s_a, n=self.config.loss_pyramid_kernel)[::2] # prob not much difference for audio reg kernel size
                inc = len(s_a) / len(s_x)
                i_a = np.round(np.arange(len(s_x)) * inc).astype(int)
                padded_s_a = nn.functional.pad(s_a[None], (0, 0, 0, 3), mode='reflect')[0]
                s_a_stacked = torch.stack([padded_s_a[i:i + n_pts_melspec] for i in i_a[:-n_pts_coord + 1]])
                s_x_stacked = torch.stack([s_x[i:i + n_pts_coord] for i in range(len(s_x) - n_pts_coord + 1)])
                stacked_list.append((s_x, s_a_stacked, s_x_stacked))
                a_new.append(s_a)
                        
            x, stacked_a, stacked_x = [torch.stack(_) for _ in zip(*stacked_list)]
            a = a_new
            if str(lvl) in keys:
                loss += self.lip_sync_loss(stacked_x, stacked_a, self.lip_syncer[str(lvl)])
                n_avg += 1

        return loss, n_avg
    

    def lip_sync_loss(self, x, a, lip_syncer):
        e_x, e_a = lip_syncer(x.flatten(end_dim=1), a.flatten(end_dim=1))
        alignment = (e_a * e_x).sum(dim=-1)
        return -alignment.mean()
    
    
    def save(self, out_dir, loader=None, val_loader=None, steps=None, epoch=None, new_file=False, best='none'):

        if new_file and (epoch is not None):
            save_dict = dict(
                checkpoints=self.dynamical_model.state_dict(),
                dis_checkpoints=self.dis.state_dict(),
                dis_optimizer=self.optim_D.state_dict(),
                gen_optimizer=self.optim_G.state_dict(),
                dis_scheduler=self.dis_scheduler.state_dict(),
                gen_scheduler=self.gen_scheduler.state_dict(),
                loader=loader,
                val_loader=val_loader,
                steps=steps,
                epoch=epoch
            )
            torch.save(save_dict, os.path.join(out_dir, f'model_chkpt_{epoch}.pt'))
        elif best != 'none':
            save_dict = dict(
                checkpoints=self.dynamical_model.state_dict(),
                steps=steps
            )
            torch.save(save_dict, os.path.join(out_dir, f'model_chkpt_best_{best}.pt'))
        elif loader is None:
            save_dict = dict(
                checkpoints=self.dynamical_model.state_dict(),
                steps=steps
            )
            torch.save(save_dict, os.path.join(out_dir, f'model_chkpt_{steps}.pt'))
        else:
            save_dict = dict(
                checkpoints=self.dynamical_model.state_dict(),
                dis_checkpoints=self.dis.state_dict(),
                dis_optimizer=self.optim_D.state_dict(),
                gen_optimizer=self.optim_G.state_dict(),
                dis_scheduler=self.dis_scheduler.state_dict(),
                gen_scheduler=self.gen_scheduler.state_dict(),
                loader=loader,
                val_loader=val_loader,
                steps=steps,
                epoch=epoch
            )
            torch.save(save_dict, os.path.join(out_dir, f'model_chkpt.pt'))

            
    def resume(self, out_dir):

        save_dict = torch.load(os.path.join(out_dir, 'model_chkpt.pt'))
        self.dynamical_model.load_state_dict(save_dict['checkpoints'])
        self.dis.load_state_dict(save_dict['dis_checkpoints'])
        self.optim_D.load_state_dict(save_dict['dis_optimizer'])
        self.optim_G.load_state_dict(save_dict['gen_optimizer'])

        dis_scheduler_state_dict = save_dict['dis_scheduler']
        dis_scheduler_state_dict['step_size'] = self.config.step_iter_lr_D
        self.dis_scheduler.load_state_dict(dis_scheduler_state_dict)
        gen_scheduler_state_dict = save_dict['gen_scheduler']
        gen_scheduler_state_dict['step_size'] = self.config.step_iter_lr
        self.gen_scheduler.load_state_dict(gen_scheduler_state_dict)

        return save_dict['loader'], save_dict['val_loader'], save_dict['epoch'], save_dict['steps']


    def step_scheduler(self):

        self.gen_scheduler.step()
        self.dis_scheduler.step()

        return self.gen_scheduler.get_last_lr()[0], self.dis_scheduler.get_last_lr()[0]
