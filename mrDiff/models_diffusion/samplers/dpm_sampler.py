"""SAMPLING ONLY."""

import torch

from .dpm_solver import NoiseScheduleVP, model_wrapper, DPM_Solver


class DPMSolverSampler(object):
    """
    DPMSolverSampler is a class for sampling using the DPM-Solver algorithm.
    Attributes:
        model (torch.nn.Module): The model to be used for sampling.
        diffusion_worker (object): An object that contains diffusion-related parameters and methods.
    Methods:
        __init__(model, diffusion_worker, **kwargs):
            Initializes the DPMSolverSampler with the given model and diffusion_worker.
        register_buffer(name, attr):
            Registers a buffer attribute to the class, ensuring it is on the correct device.
        sample(S, batch_size, shape, conditioning=None, callback=None, normals_sequence=None, img_callback=None, 
               quantize_x0=False, eta=0., mask=None, x0=None, temperature=1., noise_dropout=0., score_corrector=None, 
               corrector_kwargs=None, verbose=True, x_T=None, log_every_t=100, unconditional_guidance_scale=1., 
               unconditional_conditioning=None, **kwargs):
            Samples from the model using the DPM-Solver algorithm.
            Args:
                S (int): Number of sampling steps.
                batch_size (int): Size of the batch.
                shape (tuple): Shape of the output tensor.
                conditioning (optional): Conditioning information for the model.
                callback (optional): Callback function during sampling.
                normals_sequence (optional): Sequence of normal distributions.
                img_callback (optional): Callback function for image processing.
                quantize_x0 (bool, optional): Whether to quantize x0. Default is False.
                eta (float, optional): Noise scale. Default is 0.
                mask (optional): Mask tensor.
                x0 (optional): Initial tensor.
                temperature (float, optional): Sampling temperature. Default is 1.
                noise_dropout (float, optional): Dropout rate for noise. Default is 0.
                score_corrector (optional): Function for score correction.
                corrector_kwargs (optional): Arguments for the score corrector.
                verbose (bool, optional): Whether to print verbose messages. Default is True.
                x_T (optional): Initial tensor for sampling.
                log_every_t (int, optional): Logging frequency. Default is 100.
                unconditional_guidance_scale (float, optional): Scale for unconditional guidance. Default is 1.
                unconditional_conditioning (optional): Unconditional conditioning information.
                **kwargs: Additional arguments.
            Returns:
                torch.Tensor: The sampled tensor.
                None: Placeholder for additional return values.
    """
    def __init__(self, model, diffusion_worker, **kwargs):
        super().__init__()
        self.model = model
        self.diffusion_worker = diffusion_worker
        to_torch = lambda x: x.clone().detach().to(torch.float32) #.to(self.diffusion_worker.device)
        self.register_buffer('alphas_cumprod', to_torch(self.diffusion_worker.alphas_cumprod))

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor and attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            elif conditioning.shape[0] != batch_size:
                print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        # sampling
        F, L = shape
        size = (batch_size, F, L)

        # print(f'Data shape for DPM-Solver sampling is {size}, sampling steps {S}')

        device = self.diffusion_worker.betas.device
        if x_T is None:
            img = torch.randn(size, device=device)
        else:
            img = x_T.to(device)

        ns = NoiseScheduleVP('discrete', alphas_cumprod=self.alphas_cumprod)

        model_fn = model_wrapper(
            lambda x, t, c: self.model.forward(x, t, c),
            ns,
            model_type="x_start",
            guidance_type="classifier-free",
            condition=conditioning,
            unconditional_condition=unconditional_conditioning,
            guidance_scale=unconditional_guidance_scale,
        )

        dpm_solver = DPM_Solver(model_fn, ns, predict_x0=True, thresholding=False)
        x = dpm_solver.sample(img, steps=S, skip_type="time_uniform", method="multistep", order=2, lower_order_final=True)

        return x.to(device), None



        