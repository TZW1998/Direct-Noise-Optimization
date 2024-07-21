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
import numpy as np


# Local Model Path: Change to the model path to your own model path
CLIP_PATH = '/mnt/workspace/workgroup/tangzhiwei.tzw/clip-vit-large-patch14'
AESTHETIC_PATH = '/mnt/workspace/workgroup/tangzhiwei.tzw/reward_optimization/reward_opt/assets/sac+logos+ava1-l14-linearMSE.pth'
HPS_V2_PATH = "/mnt/workspace/workgroup/tangzhiwei.tzw/HPS_v2_compressed.pt"
PICK_SCORE_PATH = "/mnt/workspace/workgroup/tangzhiwei.tzw/pickscore"

# Aesthetic Scorer
class MLPDiff(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )


    def forward(self, embed):
        return self.layers(embed)

# 

class AestheticScorerDiff(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(CLIP_PATH)
        self.mlp = MLPDiff()
        state_dict = torch.load(AESTHETIC_PATH)
        self.mlp.load_state_dict(state_dict)
        self.dtype = dtype
        self.eval()

    def __call__(self, images):
        device = next(self.parameters()).device
        embed = self.clip.get_image_features(pixel_values=images)
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return self.mlp(embed).squeeze(1)

def aesthetic_loss_fn(device=None,
                     inference_dtype=None):
    
    target_size = 224
    normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                std=[0.26862954, 0.26130258, 0.27577711])
    scorer = AestheticScorerDiff(dtype=inference_dtype).to(device)
    scorer.requires_grad_(False)
    target_size = 224
    def loss_fn(im_pix_un, prompts=None):
        im_pix = ((im_pix_un / 2) + 0.5).clamp(0, 1) 
        im_pix = torchvision.transforms.Resize(target_size)(im_pix)
        im_pix = normalize(im_pix).to(im_pix_un.dtype)
        rewards = scorer(im_pix)
        loss = -1 * rewards

        return loss
        
    return loss_fn

def white_loss_fn(device=None,
                     inference_dtype=None):
    
    def loss_fn(im_pix_un, prompts=None):
        
        rewards = im_pix_un.mean() 
        loss = -1 * rewards

        return loss
        
    return loss_fn


def black_loss_fn(device=None,
                     inference_dtype=None):
    
    def loss_fn(im_pix_un, prompts=None):
        
        rewards = im_pix_un.mean() 
        loss =  rewards

        return loss
        
    return loss_fn

def contrast_loss_fn(device=None,
                     inference_dtype=None):
    
    def loss_fn(im_pix_un, prompts=None):
        
        rewards = im_pix_un.sum(dim=1).var()
        loss = -1 * rewards

        return loss
        
    return loss_fn

# HPS-v2
def hps_loss_fn(inference_dtype=None, device=None):
    import hpsv2
    from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer

    model_name = "ViT-H-14"
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        model_name,
        "/mnt/workspace/workgroup/tangzhiwei.tzw/open_clip_pytorch_model.bin",
        precision=inference_dtype,
        device=device,
        jit=False,
        force_quick_gelu=False,
        force_custom_text=False,
        force_patch_dropout=False,
        force_image_size=None,
        pretrained_image=False,
        image_mean=None,
        image_std=None,
        light_augmentation=True,
        aug_cfg={},
        output_dict=True,
        with_score_predictor=False,
        with_region_predictor=False
    )    
    
    tokenizer = get_tokenizer(model_name)
    
    checkpoint_path = HPS_V2_PATH
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    tokenizer = get_tokenizer(model_name)
    model = model.to(device, dtype=inference_dtype)
    model.eval()

    target_size =  224
    normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                std=[0.26862954, 0.26130258, 0.27577711])
        
    def loss_fn(im_pix_un, prompts):    
        im_pix = ((im_pix_un / 2) + 0.5).clamp(0, 1) 
        x_var = torchvision.transforms.Resize(target_size)(im_pix)
        x_var = normalize(x_var).to(im_pix.dtype)        
        caption = tokenizer(prompts)
        caption = caption.to(device)
        outputs = model(x_var, caption)
        image_features, text_features = outputs["image_features"], outputs["text_features"]
        logits = image_features @ text_features.T
        scores = torch.diagonal(logits)
        loss = - scores
        return  loss
    
    return loss_fn

# pickscore
def pick_loss_fn(inference_dtype=None, device=None):
    from open_clip import get_tokenizer

    model_name = "ViT-H-14"
    model = AutoModel.from_pretrained(PICK_SCORE_PATH) 
    
    tokenizer = get_tokenizer(model_name)
    model = model.to(device, dtype=inference_dtype)
    model.eval()

    target_size =  224
    normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                std=[0.26862954, 0.26130258, 0.27577711])
        
    def loss_fn(im_pix_un, prompts):    
        im_pix = ((im_pix_un / 2) + 0.5).clamp(0, 1) 
        x_var = torchvision.transforms.Resize(target_size)(im_pix)
        x_var = normalize(x_var).to(im_pix.dtype)        
        caption = tokenizer(prompts)
        caption = caption.to(device)
        image_embs = model.get_image_features(x_var)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
    
        text_embs = model.get_text_features(caption)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
        # score
        scores = model.logit_scale.exp() * (text_embs @ image_embs.T)[0][0]
        loss = - scores
        return  loss
    
    return loss_fn


# CLIP score evaluation
def clip_score(inference_dtype=None, device=None):
    from transformers import CLIPProcessor, CLIPModel

    model = CLIPModel.from_pretrained(CLIP_PATH)
    processor = CLIPProcessor.from_pretrained(CLIP_PATH)
    
    model = model.to(device = device, dtype=inference_dtype)
    
    def loss_fn(image, prompt):    
        inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True)
        
        for key, value in inputs.items():
            inputs[key] = value.to(device)

        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image 
        score = logits_per_image.cpu().numpy()[0][0]
        
        return  score
    
    return loss_fn

def jpeg_compressibility(inference_dtype=None, device=None):
    import io
    import numpy as np
    def loss_fn(im_pix_un, prompts):
        images = ((im_pix_un / 2) + 0.5).clamp(0, 1)
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        return torch.tensor(sizes, dtype=inference_dtype, device=device)

    return loss_fn