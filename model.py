import torch
from torchvision import transforms
from PIL import Image
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
import os.path as osp
from dassl.config import get_cfg_default

from trainers.maple import CustomCLIP, load_clip_to_cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 1. Load model
class Regressor(nn.Module):
    def __init__(self, feature_extractor='dino', hidden_dim=256, device=device, tune_blocks = []):
        super(Regressor, self).__init__()
        self.device = device
        self.feature_extractor_name = feature_extractor
        train_encoder = False
        if feature_extractor == 'dino':
            self.feature_extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').eval().to(device)
            self.input_dim = 384  # For dinov2_vits14
        elif feature_extractor == 'clip':
            import clip
            self.feature_extractor, _ = clip.load("RN50", device=device)
            self.feature_extractor.eval()
            self.input_dim = 1024  # CLIP ViT-B/32 output dim
            # Freeze all CLIP parameters
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

            # Unfreeze last transformer block of the visual encoder
            # ViT-B/32 has 12 layers; unfreeze the 12th (index -1)
            # Unfreeze only specified layers
            for name in tune_blocks:
                layer = getattr(self.feature_extractor.visual, name)
                for param in layer.parameters():
                    param.requires_grad = True

        else:
            raise ValueError(f"Unsupported feature extractor: {feature_extractor}")

        self.regressor = nn.Sequential(
        
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    @torch.no_grad()
    def extract_features(self, images):
        if self.feature_extractor_name == 'dino':
            return self.feature_extractor.forward_features(images)['x_norm_clstoken']
        elif self.feature_extractor_name == 'clip':
            images = transforms.Resize(224)(images)  # Ensure proper size
            images = transforms.CenterCrop(224)(images)
            return self.feature_extractor.encode_image(images)
    
    def forward(self, images):
        B, N, C, H, W = images.shape  # For batch processing of multiple images per sample
        images = images.view(B * N, C, H, W).to(self.device)

        features = self.extract_features(images)

        out = self.regressor(features.float())
        # out = out.view(B, N)  # Return per-sample predictions if needed
        return out.squeeze()
    


# @TRAINER_REGISTRY.register()
class MaPLe(nn.Module):
    def __init__(self):
        super(MaPLe, self).__init__()
        self.device = device
        self.cfg = setup_cfg()
        self.build_model(self.cfg)

    def build_model(self, cfg):
        classnames = ['light', 'heavy']

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.MAPLE.PREC == "fp32" or cfg.TRAINER.MAPLE.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        for name, param in self.model.named_parameters():
            # if 'transformer.resblocks.11' in name:
            if name_to_update not in name:
                param.requires_grad = False
        for name, param in self.model.named_parameters():
            if 'transformer.resblocks' in name:
                param.requires_grad = True
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                # else:
                #     param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
    
    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)


def setup_cfg():

   
    cfg = get_cfg_default()
    extend_cfg(cfg)

    cfg.merge_from_file("./configs/trainers/MaPLe/vit_b16_c2_ep5_batch4_2ctx.yaml")

    cfg.freeze()
    return cfg

def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp16"  # fp16, fp32, amp

    # Config for MaPLe
    cfg.TRAINER.MAPLE = CN()
    cfg.TRAINER.MAPLE.N_CTX = 100  # number of context vectors
    cfg.TRAINER.MAPLE.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.MAPLE.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.MAPLE.PROMPT_DEPTH = 9 # Max 12, minimum 0, for 1 it will act as shallow MaPLe (J=1)
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    # Config for independent Vision Language prompting (independent-vlp)
    cfg.TRAINER.IVLP = CN()
    cfg.TRAINER.IVLP.N_CTX_VISION = 2  # number of context vectors at the vision branch
    cfg.TRAINER.IVLP.N_CTX_TEXT = 2  # number of context vectors at the language branch
    cfg.TRAINER.IVLP.CTX_INIT = "a photo of a"  # initialization words (only for language prompts)
    cfg.TRAINER.IVLP.PREC = "fp16"  # fp16, fp32, amp
    # If both variables below are set to 0, 0, will the config will degenerate to COOP model
    cfg.TRAINER.IVLP.PROMPT_DEPTH_VISION = 9 # Max 12, minimum 0, for 0 it will act as shallow MaPLe (J=1)
    cfg.TRAINER.IVLP.PROMPT_DEPTH_TEXT = 9  # Max 12, minimum 0, for 0 it will act as shallow MaPLe (J=1)
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    # Config for only vision side prompting
    cfg.TRAINER.VPT = CN()
    cfg.TRAINER.VPT.N_CTX_VISION = 2  # number of context vectors at the vision branch
    cfg.TRAINER.VPT.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.VPT.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.VPT.PROMPT_DEPTH_VISION = 1  # if set to 1, will represent shallow vision prompting only
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new



def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head
