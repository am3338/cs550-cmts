import math
import torch
import torch.nn.functional as F
import numpy as np

from torch import nn
from einops import reduce
from tqdm.auto import tqdm
from functools import partial
from Models.autoregressive_diffusion.linear import Linear, timesteps
from Models.autoregressive_diffusion.model_utils import default, identity, extract


# gaussian diffusion trainer class

pred_len = 96

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class ARMD(nn.Module):
    def __init__(
            self,
            seq_length,
            feature_size,
            n_layer_enc=3,
            n_layer_dec=6,
            d_model=None,
            timesteps=1000,
            sampling_timesteps=None,
            loss_type='l1',
            beta_schedule='cosine',
            n_heads=4,
            mlp_hidden_times=4,
            eta=0.,
            attn_pd=0.,
            resid_pd=0.,
            w_grad=True,
            **kwargs
    ):
        super(ARMD, self).__init__()

        self.eta = eta
        self.seq_length = seq_length
        self.feature_size = feature_size

        self.model = Linear(n_feat=feature_size, n_channel=seq_length, w_grad=w_grad, **kwargs)
        #self.model_student = Linear(n_feat=feature_size, n_channel=seq_length, w_grad=w_grad, **kwargs)
        self.model_teacher = Linear(n_feat=feature_size, n_channel=seq_length, w_grad=w_grad, **kwargs)

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(
            sampling_timesteps, timesteps)  # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.fast_sampling = self.sampling_timesteps < timesteps

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate reweighting
        
        register_buffer('loss_weight', torch.sqrt(alphas) * torch.sqrt(1. - alphas_cumprod) / betas / 100)

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def output(self, x, t, training=False):
        model_output = self.model(x, t, training=training)
        return model_output

    def output_student(self, x, t, training=True):
        model_output_student = self.model(x, t, training=training)
        return model_output_student

    def output_teacher(self, x, t, training=False):
        model_output_teacher = self.model_teacher(x, t, training=training)
        return model_output_teacher

    def model_predictions(self, x, t, clip_x_start=False, training=False):
        if training:
            training = False       #padding masks = 1
        maybe_clip = partial(torch.clamp, min=-2, max=2) if clip_x_start else identity
        x_start = self.output(x, t, training) #Student output
        #x_start = maybe_clip(x_start)
        pred_noise = self.predict_noise_from_start(x, t, x_start)
        return pred_noise, x_start

    def p_mean_variance(self, x, t, clip_denoised=True):
        _, x_start = self.model_predictions(x, t)
        if clip_denoised:
            x_start.clamp_(-1., 1.)
        model_mean, posterior_variance, posterior_log_variance = \
            self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    def p_sample(self, x, t: int, clip_denoised=True):
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x=x, t=batched_times, clip_denoised=clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start


    #def get_timesteps(self, num_timesteps, num_steps):


    @torch.no_grad()
    def consistency_sample_1step(self, x, num_steps=1):
        device = self.betas.device
        shape = x.shape
        img = x[:, :pred_len, :]
        img, _ = self.p_sample(img, self.num_timesteps-1)
        return img

    @torch.no_grad()
    def sample(self, x):
        device = self.betas.device
        shape = x.shape
        img = x[:,:pred_len,:]
        #img = torch.randn(shape, device=device)
        for t in tqdm(reversed(range(0, self.num_timesteps)),
                      desc='sampling loop time step', total=self.num_timesteps):
            img, _ = self.p_sample(img, t)
        return img

    @torch.no_grad()
    def fast_sample(self, x, clip_denoised=True):
        shape = x.shape
        batch, device, total_timesteps, sampling_timesteps, eta = \
            shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.eta

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)

        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        #img = torch.randn(shape, device=device)
        img = x[:,:pred_len,:]

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, clip_x_start=clip_denoised)
            if time_next < 0:
                img = x_start
                continue
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            sigma = 0
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = 0
            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        return img

    def generate_mts(self, x):
        #print(self.fast_sampling)
        sample_fn = self.fast_sample if self.fast_sampling else self.sample
        return sample_fn(x)

    def generate_mts_cm(self, x):
        sample_fn = self.consistency_sample_1step
        return sample_fn(x)

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def q_sample(self, x_start, t, noise=None):
        index = int(t[0])+1
        x_middle = x_start[:,pred_len-index:-index,:]
        return x_middle

    def _train_loss(self, x_start, t, target=None, noise=None, training=True):
        noise = default(noise, lambda: torch.randn_like(x_start))
        if target is None:
            target = x_start[:,pred_len:,:]
        target = x_start[:,pred_len:,:]
        x = self.q_sample(x_start=x_start, t=t, noise=noise)  # noise sample
        model_out = self.output(x, t, training)
        alpha = self.sqrt_alphas_cumprod[t[0]]
        minus_alpha = self.sqrt_one_minus_alphas_cumprod[t[0]]
        target_noise = (x - target*alpha)/minus_alpha
        pred_noise = (x - model_out*alpha)/minus_alpha

        train_loss = self.loss_fn(pred_noise, target_noise, reduction='none')

        train_loss = reduce(train_loss, 'b ... -> b (...)', 'mean')
        train_loss = train_loss * extract(self.loss_weight, t, train_loss.shape)
        return train_loss.mean()

    def parameterization(self, t, T): #Let's try this simple parameterization first (it's similar to flow matching)
        c_out = t/T
        c_skip = 1 - c_out
        return c_skip, c_out

    def append_zero(self, x):
        return torch.cat([x, x.new_zeros([1])])

    def sigmas_karras(self, num_timesteps, rho=7.0, sigma_min=0.002, sigma_max=80):
        ramp = torch.linspace(0, 1, num_timesteps)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return torch.flip(self.append_zero(sigmas), dims=(0,))

    def get_probs(self, sigmas):
        probs = torch.zeros(len(sigmas)-1)
        P_mean = torch.Tensor([-1.1])
        P_std = torch.Tensor([2.0])
        for i in range(len(probs)):
            probs[i] = torch.erf(
                (torch.log(sigmas[i + 1]) - P_mean) / (
                            torch.sqrt(torch.Tensor([2])) * P_std)) - torch.erf(
                (torch.log(sigmas[i]) - P_mean) / (
                            torch.sqrt(torch.Tensor([2])) * P_std))
        probs = probs / torch.sum(probs)
        #probs = probs.numpy()
        return probs

    def cm_reweighting(self, sigmas, t_student, t_teacher):
        return 1.0/(sigmas[t_student] - sigmas[t_teacher])

    def get_index_probs(self, N): #Imputing points close to future ones can be misleading...
        scores = torch.flip(torch.arange(1,N), dims=(0,))
        #print(scores)
        scores = scores/torch.sum(scores, dim=0)
        #print(scores)
        #input()
        return scores

    def _train_loss_consistency1(self, x_start, t_student, t_teacher, num_timesteps=None, target=None, noise=None, training=True):
        noise = default(noise, lambda: torch.randn_like(x_start))
        c_skip_student, c_out_student = self.parameterization(t_student, num_timesteps)
        c_skip_teacher, c_out_teacher = self.parameterization(t_teacher, num_timesteps)
        if target is None:
            target = x_start[:, pred_len:, :]
        target = x_start[:, pred_len:, :]
        K = 8 #Set this as a hyperparameter
        probs = self.get_index_probs(pred_len - 1)
        indices = torch.multinomial(probs.expand(x_start.shape[0], -1), K, replacement=False)

        indices = torch.clamp(indices, min=1, max=pred_len-2)

        left = x_start[torch.arange(x_start.shape[0]).unsqueeze(1), indices - 1]
        right = x_start[torch.arange(x_start.shape[0]).unsqueeze(1), indices + 1]
        interpolated = 0.5 * left + 0.5 * right
        x_start_interpolated = x_start.clone()
        x_start_interpolated[torch.arange(x_start.shape[0]).unsqueeze(1), indices] = interpolated

        x_student = self.q_sample(x_start=x_start, t=t_student, noise=noise)
        x_student_interpolated = self.q_sample(x_start=x_start_interpolated, t=t_student, noise=noise)
        model_out_student = self.output_student(x_student, t_student, training=True)
        model_out_student_interpolated = self.output_student(x_student_interpolated, t_student, training=True)
        #alpha_student = self.sqrt_alphas_cumprod[t_student[0]]
        #minus_alpha_student = self.sqrt_one_minus_alphas_cumprod[t_student[0]]
        #student_noise = (x_student - model_out_student * alpha_student)/minus_alpha_student

        #@torch.no_grad()
        #def teacher_output(x, t):
        #    return self.output_teacher(x, t, training=False)

        #x_teacher = self.q_sample(x_start=x_start, t=t_teacher, noise=noise)
        #model_out_teacher = teacher_output(x_teacher, t_teacher)
        #model_out_teacher = model_out_teacher.detach()
        #alpha_teacher = self.sqrt_alphas_cumprod[t_teacher[0]]
        #minus_alpha_teacher = self.sqrt_one_minus_alphas_cumprod[t_teacher[0]]
        #teacher_noise = (x_teacher - model_out_teacher * alpha_teacher)/minus_alpha_teacher

        #target_noise = (x_student - target * alpha_student)/minus_alpha_student

        f_student = c_skip_student[:, None, None] * x_student + c_out_student[:, None, None] * model_out_student
        f_student_interpolated = c_skip_student[:, None, None] * x_student_interpolated + c_out_student[:, None, None] * model_out_student_interpolated
        #f_teacher = c_skip_teacher[:, None, None] * x_teacher + c_out_teacher[:, None, None] * model_out_teacher

        #Direct denoising objective
        denoising_loss = self.loss_fn(f_student, target, reduction='none')

        contrastive_loss = self.loss_fn(f_student, f_student_interpolated, reduction='none')

        #Index-based interpolation:
        #K = 1: MSE: 0.5060, MAE: 0.4938
        #K = 2:
        #K = 8: MSE: 0.5050, MAE: 0.4946

        train_loss = denoising_loss #+ contrastive_loss

        train_loss = reduce(train_loss, 'b ... -> b (...)', 'mean')
        self.loss_weight = t_student/num_timesteps #MSE: 0.5048 MAE: 0.4938
        #self.loss_weight = torch.ones_like(t_student) #MSE: 0.5089 MAE: 0.4943
        #Diffusion weights: MSE: 0.5155 MAE: 0.4942
        train_loss = train_loss * extract(self.loss_weight, t_student, train_loss.shape)

        #ema_rate = 0.0
        #Update teacher
        #for s_params, t_params in zip(self.model.parameters(), self.model_teacher.parameters()):
        #    t_params.detach().mul_(ema_rate).add_(s_params, alpha = 1 - ema_rate)

        return train_loss.mean()

    def _train_loss_consistency2(self, x_start, t_student, t_teacher, num_timesteps=None, target=None, noise=None, training=True):
        noise = default(noise, lambda: torch.randn_like(x_start))
        c_skip_student, c_out_student = self.parameterization(t_student, num_timesteps)
        c_skip_teacher, c_out_teacher = self.parameterization(t_teacher, num_timesteps)
        if target is None:
            target = x_start[:, pred_len:, :]
        target = x_start[:, pred_len:, :]
        #K = 8 #Set this as a hyperparameter
        #probs = self.get_index_probs(pred_len - 1)
        #indices = torch.multinomial(probs.expand(x_start.shape[0], -1), K, replacement=False)

        #indices = torch.clamp(indices, min=1, max=pred_len-2)

        #left = x_start[torch.arange(x_start.shape[0]).unsqueeze(1), indices - 1]
        #right = x_start[torch.arange(x_start.shape[0]).unsqueeze(1), indices + 1]
        #interpolated = 0.5 * left + 0.5 * right
        #x_start_interpolated = x_start.clone()
        #x_start_interpolated[torch.arange(x_start.shape[0]).unsqueeze(1), indices] = interpolated

        x_student = self.q_sample(x_start=x_start, t=t_student, noise=noise)
        #x_student_interpolated = self.q_sample(x_start=x_start_interpolated, t=t_student, noise=noise)
        model_out_student = self.output_student(x_student, t_student, training=True)
        #model_out_student_interpolated = self.output_student(x_student_interpolated, t_student, training=True)
        #alpha_student = self.sqrt_alphas_cumprod[t_student[0]]
        #minus_alpha_student = self.sqrt_one_minus_alphas_cumprod[t_student[0]]
        #student_noise = (x_student - model_out_student * alpha_student)/minus_alpha_student

        @torch.no_grad()
        def teacher_output(x, t):
            return self.output_teacher(x, t, training=False)

        x_teacher = self.q_sample(x_start=x_start, t=t_teacher, noise=noise)
        model_out_teacher = teacher_output(x_teacher, t_teacher)
        model_out_teacher = model_out_teacher.detach()
        #alpha_teacher = self.sqrt_alphas_cumprod[t_teacher[0]]
        #minus_alpha_teacher = self.sqrt_one_minus_alphas_cumprod[t_teacher[0]]
        #teacher_noise = (x_teacher - model_out_teacher * alpha_teacher)/minus_alpha_teacher

        #target_noise = (x_student - target * alpha_student)/minus_alpha_student

        f_student = c_skip_student[:, None, None] * x_student + c_out_student[:, None, None] * model_out_student
        #f_student_interpolated = c_skip_student[:, None, None] * x_student_interpolated + c_out_student[:, None, None] * model_out_student_interpolated
        f_teacher = c_skip_teacher[:, None, None] * x_teacher + c_out_teacher[:, None, None] * model_out_teacher

        #Direct denoising objective
        denoising_loss = self.loss_fn(f_student, f_teacher, reduction='none')

        #contrastive_loss = self.loss_fn(f_student, f_student_interpolated, reduction='none')


        train_loss = denoising_loss #+ contrastive_loss

        train_loss = reduce(train_loss, 'b ... -> b (...)', 'mean')
        self.loss_weight = t_student/num_timesteps #MSE: 0.5048 MAE: 0.4938
        #self.loss_weight = torch.ones_like(t_student) #MSE: 0.5089 MAE: 0.4943
        #Diffusion weights: MSE: 0.5155 MAE: 0.4942
        train_loss = train_loss * extract(self.loss_weight, t_student, train_loss.shape)

        #ema_rate = 0.0
        #Update teacher
        #for s_params, t_params in zip(self.model.parameters(), self.model_teacher.parameters()):
        #    t_params.detach().mul_(ema_rate).add_(s_params, alpha = 1 - ema_rate)

        return train_loss.mean()

    def forward(self, x, **kwargs):
        b, c, n, device, feature_size, = *x.shape, x.device, self.feature_size
        assert n == feature_size, f'number of variable must be {feature_size}'
        t_student = torch.randint(1, self.num_timesteps, (1,), device=device).repeat(b).long() #Setting t_student=1 results in NaN loss
        t_teacher = t_student - 1
        #t = torch.randint(0, self.num_timesteps, (1,), device=device).repeat(b).long()
        #return self._train_loss(x_start=x, t=t)
        #print(t_student)
        #print(t_teacher)
        return self._train_loss_consistency1(x_start=x, t_student=t_student, t_teacher=t_teacher, num_timesteps=self.num_timesteps, **kwargs)

    def langevin_fn(
        self,
        coef,
        partial_mask,
        tgt_embs,
        learning_rate,
        sample,
        mean,
        sigma,
        t,
        coef_=0.
    ):
    
        if t[0].item() < self.num_timesteps * 0.05:
            K = 0
        elif t[0].item() > self.num_timesteps * 0.9:
            K = 3
        elif t[0].item() > self.num_timesteps * 0.75:
            K = 2
            learning_rate = learning_rate * 0.5
        else:
            K = 1
            learning_rate = learning_rate * 0.25

        input_embs_param = torch.nn.Parameter(sample)

        with torch.enable_grad():
            for i in range(K):
                optimizer = torch.optim.Adagrad([input_embs_param], lr=learning_rate)
                optimizer.zero_grad()

                x_start = self.output(x=input_embs_param, t=t)

                if sigma.mean() == 0:
                    logp_term = coef * ((mean - input_embs_param) ** 2 / 1.).mean(dim=0).sum()
                    infill_loss = (x_start[partial_mask] - tgt_embs[partial_mask]) ** 2
                    infill_loss = infill_loss.mean(dim=0).sum()
                else:
                    logp_term = coef * ((mean - input_embs_param)**2 / sigma).mean(dim=0).sum()
                    infill_loss = (x_start[partial_mask] - tgt_embs[partial_mask]) ** 2
                    infill_loss = (infill_loss/sigma.mean()).mean(dim=0).sum()
            
                loss = logp_term + infill_loss
                loss.backward()
                optimizer.step()
                epsilon = torch.randn_like(input_embs_param.data)
                input_embs_param = torch.nn.Parameter((input_embs_param.data + coef_ * sigma.mean().item() * epsilon).detach())

        sample[~partial_mask] = input_embs_param.data[~partial_mask]
        return sample
    

if __name__ == '__main__':
    pass
