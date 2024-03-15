import torch
import torch.nn as nn
import torch.nn.functional as F

def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

class LSD(nn.Module):
    def __init__(self, model_ae, model_se, model_dif, args,
                 mean_type='epsilon', var_type='fixedlarge'):
        super().__init__()
        
        self.device = args.device
        
        self.model_ae = model_ae
        self.model_se = model_se
        self.model_dif = model_dif
        
        self.T = args.dif_T
        self.scaling = args.dif_scaling
        self.register_buffer(
            'betas', torch.linspace(args.dif_beta_1, args.dif_beta_T, args.dif_T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

        self.mean_type = mean_type
        self.var_type = var_type
        
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:args.dif_T]

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_recip_alphas_bar', torch.sqrt(1. / alphas_bar))
        self.register_buffer(
            'sqrt_recipm1_alphas_bar', torch.sqrt(1. / alphas_bar - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer(
            'posterior_var',
            self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
        # below: log calculation clipped because the posterior variance is 0 at
        # the beginning of the diffusion chain
        self.register_buffer(
            'posterior_log_var_clipped',
            torch.log(
                torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])))
        self.register_buffer(
            'posterior_mean_coef1',
            torch.sqrt(alphas_bar_prev) * self.betas / (1. - alphas_bar))
        self.register_buffer(
            'posterior_mean_coef2',
            torch.sqrt(alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar))
        
    def forward(self, image):
        latent = self.model_ae.encode(image).sample() * self.scaling
        slots = self.model_se(image)
        t = torch.randint(self.T, size=(latent.shape[0], )).to(self.device)
        noise = torch.randn_like(latent)
        latent_t = (
            extract(self.sqrt_alphas_bar, t, latent.shape) * latent +
            extract(self.sqrt_one_minus_alphas_bar, t, latent.shape) * noise)
        pred = self.model_dif(latent_t, t, slots)
        loss = nn.MSELoss()(pred, noise)
        return loss
    
    def q_mean_variance(self, x_0, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior
        q(x_{t-1} | x_t, x_0)
        """
        assert x_0.shape == x_t.shape
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_var_clipped = extract(
            self.posterior_log_var_clipped, t, x_t.shape)
        return posterior_mean, posterior_log_var_clipped

    def predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps
        )

    def predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            extract(
                1. / self.posterior_mean_coef1, t, x_t.shape) * xprev -
            extract(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t,
                x_t.shape) * x_t
        )

    def p_mean_variance(self, x_t, t, slots):
        # below: only log_variance is used in the KL computations
        model_log_var = {
            # for fixedlarge, we set the initial (log-)variance like so to
            # get a better decoder log likelihood
            'fixedlarge': torch.log(torch.cat([self.posterior_var[1:2],
                                               self.betas[1:]])),
            'fixedsmall': self.posterior_log_var_clipped,
        }[self.var_type]
        model_log_var = extract(model_log_var, t, x_t.shape)

        # Mean parameterization
        if self.mean_type == 'xprev':       # the model predicts x_{t-1}
            x_prev = self.model_dif(x_t, t, slots)
            x_0 = self.predict_xstart_from_xprev(x_t, t, xprev=x_prev)
            model_mean = x_prev
        elif self.mean_type == 'xstart':    # the model predicts x_0
            x_0 = self.model_dif(x_t, t, slots)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        elif self.mean_type == 'epsilon':   # the model predicts epsilon
            eps = self.model_dif(x_t, t, slots)
            x_0 = self.predict_xstart_from_eps(x_t, t, eps=eps)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        else:
            raise NotImplementedError(self.mean_type)

        return model_mean, model_log_var

    def sample(self, x_T, slots):
        """
        Algorithm 2.
        """
        x_t = x_T
        for time_step in reversed(range(self.T)):
            print(f"Time step: {time_step}")
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, log_var = self.p_mean_variance(x_t=x_t, t=t, slots=slots)
            # no noise when t == 0
            if time_step > 0:
                noise = torch.randn_like(x_t) * self.scaling
            else:
                noise = 0
            x_t = mean + torch.exp(0.5 * log_var) * noise
        x_0 = x_t
        return x_0
    def recon(self, image):
        slots = self.model_se(image)
        latent = self.model_ae.encode(image).sample() * self.scaling
        noise = torch.randn_like(latent) #* self.scaling
        t = latent.new_ones([latent.shape[0], ], dtype=torch.long) * (self.T - 1)
        latent_T = (
            extract(self.sqrt_alphas_bar, t, latent.shape) * latent +
            extract(self.sqrt_one_minus_alphas_bar, t, latent.shape) * noise)
        latent_0 = self.sample(latent_T, slots) / self.scaling
        pred = self.model_ae.decode(latent_0)
        return pred