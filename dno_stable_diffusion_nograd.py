import torch
import torch.nn as nn
import torchvision
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
import argparse
import torch.utils.checkpoint as checkpoint
import os, shutil
from PIL import Image
import time
from torch import autocast
from torch.cuda.amp import GradScaler
from transformers import CLIPModel, CLIPProcessor, AutoProcessor, AutoModel
from accelerate import Accelerator
import numpy as np
from rewards import RFUNCTIONS
import json


# sampling algorithm
class SequentialDDIM:

    def __init__(self, timesteps = 100, scheduler = None, eta = 0.0, cfg_scale = 4.0, device = "cuda", opt_timesteps = 50):
        self.eta = eta 
        self.timesteps = timesteps
        self.num_steps = timesteps
        self.scheduler = scheduler
        self.device = device
        self.cfg_scale = cfg_scale
        self.opt_timesteps = opt_timesteps 

        # compute some coefficients in advance
        scheduler_timesteps = self.scheduler.timesteps.tolist()
        scheduler_prev_timesteps = scheduler_timesteps[1:]
        scheduler_prev_timesteps.append(0)
        self.scheduler_timesteps = scheduler_timesteps[::-1]
        scheduler_prev_timesteps = scheduler_prev_timesteps[::-1]
        alphas_cumprod = [1 - self.scheduler.alphas_cumprod[t] for t in self.scheduler_timesteps]
        alphas_cumprod_prev = [1 - self.scheduler.alphas_cumprod[t] for t in scheduler_prev_timesteps]

        now_coeff = torch.tensor(alphas_cumprod)
        next_coeff = torch.tensor(alphas_cumprod_prev)
        now_coeff = torch.clamp(now_coeff, min = 0)
        next_coeff = torch.clamp(next_coeff, min = 0)
        m_now_coeff = torch.clamp(1 - now_coeff, min = 0)
        m_next_coeff = torch.clamp(1 - next_coeff, min = 0)
        self.noise_thr = torch.sqrt(next_coeff / now_coeff) * torch.sqrt(1 - (1 - now_coeff) / (1 - next_coeff))
        self.nl = self.noise_thr * self.eta
        self.nl[0] = 0.
        m_nl_next_coeff = torch.clamp(next_coeff - self.nl**2, min = 0)
        self.coeff_x = torch.sqrt(m_next_coeff) / torch.sqrt(m_now_coeff)
        self.coeff_d = torch.sqrt(m_nl_next_coeff) - torch.sqrt(now_coeff) * self.coeff_x

    def is_finished(self):
        return self._is_finished

    def get_last_sample(self):
        return self._samples[0]

    def prepare_model_kwargs(self, prompt_embeds = None):

        t_ind = self.num_steps - len(self._samples)
        t = self.scheduler_timesteps[t_ind]
   
        model_kwargs = {
            "sample": torch.stack([self._samples[0], self._samples[0]]),
            "timestep": torch.tensor([t, t], device = self.device),
            "encoder_hidden_states": prompt_embeds
        }

        model_kwargs["sample"] = self.scheduler.scale_model_input(model_kwargs["sample"],t)

        return model_kwargs


    def step(self, model_output):
        model_output_uncond, model_output_text = model_output[0].chunk(2)
        direction = model_output_uncond + self.cfg_scale * (model_output_text - model_output_uncond)
        direction = direction[0]

        t = self.num_steps - len(self._samples)

        if t <= self.opt_timesteps:
            now_sample = self.coeff_x[t] * self._samples[0] + self.coeff_d[t] * direction  + self.nl[t] * self.noise_vectors[t]
        else:
            with torch.no_grad():
                now_sample = self.coeff_x[t] * self._samples[0] + self.coeff_d[t] * direction  + self.nl[t] * self.noise_vectors[t]

        self._samples.insert(0, now_sample)
        
        if len(self._samples) > self.timesteps:
            self._is_finished = True

    def initialize(self, noise_vectors):
        self._is_finished = False

        self.noise_vectors = noise_vectors

        if self.num_steps == self.opt_timesteps:
            self._samples = [self.noise_vectors[-1]]
        else:
            self._samples = [self.noise_vectors[-1].detach()]

def sequential_sampling(pipeline, unet, sampler, prompt_embeds, noise_vectors): 


    sampler.initialize(noise_vectors)

    model_time = 0
    while not sampler.is_finished():
        model_kwargs = sampler.prepare_model_kwargs(prompt_embeds = prompt_embeds)
        #model_output = pipeline.unet(**model_kwargs)
        model_output = checkpoint.checkpoint(unet, model_kwargs["sample"], model_kwargs["timestep"], model_kwargs["encoder_hidden_states"],  use_reentrant=False)
        sampler.step(model_output) 

    return sampler.get_last_sample()

# sampling algorithm
class BatchSequentialDDIM:

    def __init__(self, timesteps = 100, scheduler = None, eta = 0.0, cfg_scale = 4.0, device = "cuda", opt_timesteps = 50):
        self.eta = eta 
        self.timesteps = timesteps
        self.num_steps = timesteps
        self.scheduler = scheduler
        self.device = device
        self.cfg_scale = cfg_scale
        self.opt_timesteps = opt_timesteps 

        # compute some coefficients in advance
        scheduler_timesteps = self.scheduler.timesteps.tolist()
        scheduler_prev_timesteps = scheduler_timesteps[1:]
        scheduler_prev_timesteps.append(0)
        self.scheduler_timesteps = scheduler_timesteps[::-1]
        scheduler_prev_timesteps = scheduler_prev_timesteps[::-1]
        alphas_cumprod = [1 - self.scheduler.alphas_cumprod[t] for t in self.scheduler_timesteps]
        alphas_cumprod_prev = [1 - self.scheduler.alphas_cumprod[t] for t in scheduler_prev_timesteps]

        now_coeff = torch.tensor(alphas_cumprod)
        next_coeff = torch.tensor(alphas_cumprod_prev)
        now_coeff = torch.clamp(now_coeff, min = 0)
        next_coeff = torch.clamp(next_coeff, min = 0)
        m_now_coeff = torch.clamp(1 - now_coeff, min = 0)
        m_next_coeff = torch.clamp(1 - next_coeff, min = 0)
        self.noise_thr = torch.sqrt(next_coeff / now_coeff) * torch.sqrt(1 - (1 - now_coeff) / (1 - next_coeff))
        self.nl = self.noise_thr * self.eta
        self.nl[0] = 0.
        m_nl_next_coeff = torch.clamp(next_coeff - self.nl**2, min = 0)
        self.coeff_x = torch.sqrt(m_next_coeff) / torch.sqrt(m_now_coeff)
        self.coeff_d = torch.sqrt(m_nl_next_coeff) - torch.sqrt(now_coeff) * self.coeff_x

    def is_finished(self):
        return self._is_finished

    def get_last_sample(self):
        return self._samples[0]

    def prepare_model_kwargs(self, prompt_embeds = None):

        t_ind = self.num_steps - len(self._samples)
        t = self.scheduler_timesteps[t_ind]
        batch = len(self._samples[0])

        uncond_embeds = torch.stack([prompt_embeds[0]] * batch)
        cond_embeds = torch.stack([prompt_embeds[1]] * batch)
   
        model_kwargs = {
            "sample": torch.concat([self._samples[0], self._samples[0]]),
            "timestep": torch.tensor([t] * 2 * batch, device = self.device),
            "encoder_hidden_states": torch.concat([uncond_embeds, cond_embeds])
        }

        model_kwargs["sample"] = self.scheduler.scale_model_input(model_kwargs["sample"],t)
    
        return model_kwargs


    def step(self, model_output):
        model_output_uncond, model_output_text = model_output[0].chunk(2)
        direction = model_output_uncond + self.cfg_scale * (model_output_text - model_output_uncond)

        t = self.num_steps - len(self._samples)

        if t <= self.opt_timesteps:
            now_sample = self.coeff_x[t] * self._samples[0] + self.coeff_d[t] * direction  + self.nl[t] * self.noise_vectors[t]
        else:
            with torch.no_grad():
                now_sample = self.coeff_x[t] * self._samples[0] + self.coeff_d[t] * direction  + self.nl[t] * self.noise_vectors[t]

        self._samples.insert(0, now_sample)
        
        if len(self._samples) > self.timesteps:
            self._is_finished = True

    def initialize(self, noise_vectors):
        self._is_finished = False

        self.noise_vectors = noise_vectors

        self._samples = [self.noise_vectors[-1]]
  

def batch_sequential_sampling(pipeline, unet, sampler, prompt_embeds, noise_vectors): 


    sampler.initialize(noise_vectors)

    model_time = 0
    while not sampler.is_finished():
        model_kwargs = sampler.prepare_model_kwargs(prompt_embeds = prompt_embeds)
        model_output = unet(**model_kwargs)
        sampler.step(model_output) 

    return sampler.get_last_sample()


def decode_latent(decoder, latent):
    img = decoder.decode(latent / 0.18215).sample
    return img

def to_img(img):
    img = torch.clamp(127.5 * img.cpu() + 128.0, 0, 255).permute(1, 2, 0).to(dtype=torch.uint8).numpy()

    return img

def compute_probability_regularization(noise_vectors, eta, opt_time, subsample, shuffled_times = 100):
    # squential subsampling
    if eta > 0:
        noise_vectors_flat = noise_vectors[:(opt_time + 1)].flatten()
    else:
        noise_vectors_flat = noise_vectors[-1].flatten()
        
    dim = noise_vectors_flat.shape[0]

    # use for computing the probability regularization
    subsample_dim = round(4 ** subsample)
    subsample_num = dim // subsample_dim
        
    noise_vectors_seq = noise_vectors_flat.view(subsample_num, subsample_dim)

    seq_mean = noise_vectors_seq.mean(dim = 0)
    noise_vectors_seq = noise_vectors_seq / np.sqrt(subsample_num)
    seq_cov = noise_vectors_seq.T @ noise_vectors_seq
    seq_var = seq_cov.diag()
    
    # compute the probability of the noise
    seq_mean_M = torch.norm(seq_mean)
    seq_cov_M = torch.linalg.matrix_norm(seq_cov - torch.eye(subsample_dim, device = seq_cov.device), ord = 2)
    
    seq_mean_log_prob = - (subsample_num * seq_mean_M ** 2) / 2 / subsample_dim
    seq_mean_log_prob = torch.clamp(seq_mean_log_prob, max = - np.log(2))
    seq_mean_prob = 2 * torch.exp(seq_mean_log_prob)
    seq_cov_diff = torch.clamp(torch.sqrt(1+seq_cov_M) - 1 - np.sqrt(subsample_dim/subsample_num), min = 0)
    seq_cov_log_prob = - subsample_num * (seq_cov_diff ** 2) / 2 
    seq_cov_log_prob = torch.clamp(seq_cov_log_prob, max = - np.log(2))
    seq_cov_prob = 2 * torch.exp(seq_cov_log_prob)

    shuffled_mean_prob_list = []
    shuffled_cov_prob_list = [] 
    
    shuffled_mean_log_prob_list = []
    shuffled_cov_log_prob_list = [] 
    
    shuffled_mean_M_list = []
    shuffled_cov_M_list = []

    for _ in range(shuffled_times):
        noise_vectors_flat_shuffled = noise_vectors_flat[torch.randperm(dim)]   
        noise_vectors_shuffled = noise_vectors_flat_shuffled.view(subsample_num, subsample_dim)
        
        shuffled_mean = noise_vectors_shuffled.mean(dim = 0)
        noise_vectors_shuffled = noise_vectors_shuffled / np.sqrt(subsample_num)
        shuffled_cov = noise_vectors_shuffled.T @ noise_vectors_shuffled
        shuffled_var = shuffled_cov.diag()
        
        # compute the probability of the noise
        shuffled_mean_M = torch.norm(shuffled_mean)
        shuffled_cov_M = torch.linalg.matrix_norm(shuffled_cov - torch.eye(subsample_dim, device = shuffled_cov.device), ord = 2)
        

        shuffled_mean_log_prob = - (subsample_num * shuffled_mean_M ** 2) / 2 / subsample_dim
        shuffled_mean_log_prob = torch.clamp(shuffled_mean_log_prob, max = - np.log(2))
        shuffled_mean_prob = 2 * torch.exp(shuffled_mean_log_prob)
        shuffled_cov_diff = torch.clamp(torch.sqrt(1+shuffled_cov_M) - 1 - np.sqrt(subsample_dim/subsample_num), min = 0)
        
        shuffled_cov_log_prob = - subsample_num * (shuffled_cov_diff ** 2) / 2
        shuffled_cov_log_prob = torch.clamp(shuffled_cov_log_prob, max = - np.log(2))
        shuffled_cov_prob = 2 * torch.exp(shuffled_cov_log_prob) 
        
        
        shuffled_mean_prob_list.append(shuffled_mean_prob.item())
        shuffled_cov_prob_list.append(shuffled_cov_prob.item())
        
        shuffled_mean_log_prob_list.append(shuffled_mean_log_prob)
        shuffled_cov_log_prob_list.append(shuffled_cov_log_prob)
        
        shuffled_mean_M_list.append(shuffled_mean_M.item())
        shuffled_cov_M_list.append(shuffled_cov_M.item())
        
    reg_loss = - (seq_mean_log_prob + seq_cov_log_prob + (sum(shuffled_mean_log_prob_list) + sum(shuffled_cov_log_prob_list)) / shuffled_times)
    
    return reg_loss

def main():
    parser = argparse.ArgumentParser(description='Diffusion Optimization with Differentiable Objective')
    parser.add_argument('--model', type=str, default="runwayml/stable-diffusion-v1-5", help='path to the model')
    parser.add_argument('--prompt', type=str, default="black duck", help='prompt for the optimization')
    parser.add_argument('--num_steps', type=int, default=50, help='number of steps for optimization')
    parser.add_argument('--eta', type=float, default=1.0, help='noise scale')
    parser.add_argument('--guidance_scale', type=float, default=5.0, help='guidance scale')
    parser.add_argument('--device', type=str, default="cuda", help='device for optimization')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--opt_steps', type=int, default=50, help='number of optimization steps')
    parser.add_argument('--opt_time', type=int, default=50)
    parser.add_argument('--objective', type=str, default="jpeg", help='objective for optimization', choices = ["aesthetic", "hps", "pick", "jpeg"])
    parser.add_argument('--precision', choices = ["fp16", "fp32"], default="fp16", help='precision for optimization')
    parser.add_argument('--batch_per_device', type=int, default=4, help='batch size per device')
    parser.add_argument('--mu', type=float, default=0.01, help='control the precison of gradient approxmiation')
    parser.add_argument('--gamma', type=float, default=0., help='control the precison of gradient approxmiation')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate for optimization')
    parser.add_argument('--output', type=str, default="output", help='output path')

    args = parser.parse_args()

    if args.precision == "bf16":
        inference_dtype = torch.bfloat16
    elif args.precision == "fp16":
        inference_dtype = torch.float16
    else:
        inference_dtype = torch.float32

    accelerator = Accelerator()

    main_inferece_dtype = inference_dtype

    # load model
    pipeline = StableDiffusionPipeline.from_pretrained(args.model).to(device = accelerator.device)
    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(False)
    # disable safety checker
    pipeline.safety_checker = None
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    # set the number of steps
    pipeline.scheduler.set_timesteps(args.num_steps)

    # load lora if needed
    # only do gradient computation at the main process
    # the other process are used to generate samples for approximating the gradient of reward function
    if accelerator.process_index == 0:
        inference_dtype = torch.float32
    else:
        pipeline.vae.to(dtype = inference_dtype)
        pipeline.text_encoder.to(dtype = inference_dtype)
        pipeline.unet.to(dtype = inference_dtype)
        
    unet = pipeline.unet

    # load the loss function, which is negative of the reward fucntion
    loss_dtype = torch.float32
    loss_fn = RFUNCTIONS[args.objective](inference_dtype = loss_dtype, device = args.device)

    torch.manual_seed(args.seed)
    noise_vectors = torch.randn(args.num_steps + 1, 4, 64, 64, device = accelerator.device)

    # make sure all devices have the same initial noise vectors
    noise_vectors = noise_vectors.unsqueeze(0)
    noise_vectors = accelerator.gather(noise_vectors)
    noise_vectors = noise_vectors[0]

    # make sure all devices have different randomness
    torch.manual_seed(args.seed + accelerator.process_index)
    torch.manual_seed(torch.randint(0, 1000, (1,)).item() + accelerator.process_index)

    # only do optimization at the main process
    if accelerator.process_index == 0:
        optimize_groups = []
        noise_vectors.requires_grad_(True)
        optimize_groups.append({"params":noise_vectors, "lr":args.lr, "weight_decy": 0})
        optimizer = torch.optim.AdamW(optimize_groups)
    
    prompt_embeds = pipeline._encode_prompt(
                        args.prompt,
                        accelerator.device,
                        1,
                        True,
                    ).to(dtype = inference_dtype)

    if accelerator.process_index == 0:
        path_name = f"SD-NoGrad-{time.strftime('%Y-%m-%d-%H-%M-%S')}"
        output_path = os.path.join(args.output, path_name)
        
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path)
        
        # save args
        with open(os.path.join(output_path, "args.json"), "w") as f:
            json.dump(args.__dict__, f, indent = 4)

        use_amp = False if args.precision == "fp32" else True
        grad_scaler = GradScaler(enabled=use_amp, init_scale = 8192)
        amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
        
    else:
        use_mu = args.mu
        
    

    # start optimization
    for i in range(args.opt_steps):
        
        
        if accelerator.process_index == 0:
            start_time = time.time()
            optimizer.zero_grad()
            with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                ddim_sampler = SequentialDDIM(timesteps = args.num_steps,
                                                scheduler = pipeline.scheduler, 
                                                eta = args.eta, 
                                                cfg_scale = args.guidance_scale, 
                                                device = accelerator.device,
                                                opt_timesteps = args.opt_time)

                sample = sequential_sampling(pipeline, unet, ddim_sampler, prompt_embeds = prompt_embeds, noise_vectors = noise_vectors)
                sample = decode_latent(pipeline.vae, sample.unsqueeze(0))[0]

                est_grad = torch.zeros_like(sample.detach()).to(dtype = loss_dtype)

                losses = torch.zeros(args.batch_per_device + 1, device = accelerator.device, dtype = loss_dtype)

            with torch.no_grad():
                reward = - loss_fn(sample.unsqueeze(0), [args.prompt]).item()


        else:
 
            with torch.no_grad():
                ddim_sampler = BatchSequentialDDIM(timesteps = args.num_steps,
                                                scheduler = pipeline.scheduler, 
                                                eta = args.eta, 
                                                cfg_scale = args.guidance_scale, 
                                                device = accelerator.device,
                                                opt_timesteps = args.opt_time)
                noise_vectors_flat = noise_vectors.detach().unsqueeze(1).to(dtype=inference_dtype)
                cand_noise_vectors = noise_vectors_flat + use_mu * torch.randn(args.num_steps + 1, args.batch_per_device , 4, 64, 64, device = accelerator.device, dtype = inference_dtype)
                cand_noise_vectors = torch.concat([cand_noise_vectors, noise_vectors_flat], dim = 1)

                
                samples = batch_sequential_sampling(pipeline, unet, ddim_sampler, prompt_embeds = prompt_embeds, noise_vectors = cand_noise_vectors)
                
                samples = decode_latent(pipeline.vae, samples).to(dtype = loss_dtype)

                losses = loss_fn(samples, [args.prompt] * samples.shape[0])

                est_grad = torch.zeros_like(samples[-1])
                for i in range(args.batch_per_device):
                    est_grad += (losses[i] - losses[-1]) * (samples[i] - samples[-1])

            

        est_grad = est_grad.unsqueeze(0)
        est_grad = accelerator.gather(est_grad)
        est_grad = est_grad[1:].mean(dim = 0)
        est_grad /= (torch.norm(est_grad) + 1e-3)

        losses = accelerator.gather(losses)

        if accelerator.process_index == 0:
            with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                loss = torch.sum(est_grad * sample)

                if args.gamma > 0:
                    reg_loss = compute_probability_regularization(noise_vectors, args.eta, args.opt_time, args.subsample)
                    loss = loss + args.gamma * reg_loss

                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)

                # normalize the gradient
                torch.nn.utils.clip_grad_norm_([noise_vectors], 1.0)

                grad_scaler.step(optimizer)
                grad_scaler.update()
           
            img = to_img(sample)
            img = Image.fromarray(img)
            img.save(os.path.join(output_path, f"{i}_{reward}.png"))
            print(f"step : {i}, reward : {reward}")

        # synchronize the noise vector
        noise_vectors_data = noise_vectors.data
        noise_vectors_data = noise_vectors_data.unsqueeze(0)
        noise_vectors_data = accelerator.gather(noise_vectors_data)

        if accelerator.process_index != 0:
            noise_vectors_data = noise_vectors_data[0]
            noise_vectors.data = noise_vectors_data  

if __name__ == "__main__":
    main()