import copy
import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

from scipy.spatial.distance import pdist, squareform
import numpy as np

from collections import OrderedDict
CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.clip = clip_model

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.APP.N_CTX
        ctx_init = cfg.TRAINER.APP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        self.N = cfg.TRAINER.APP.N
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        vis_dim = clip_model.visual.output_dim
        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.APP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(self.N, n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)  # define the prompt to be trained
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        ])).half()

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        tokenized_prompts = tokenized_prompts.repeat(self.N, 1)
        # tokenized_prompts3.view(3,100,77)

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

            # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.APP.CLASS_TOKEN_POSITION

    def forward(self):

        ctx = self.ctx
        if ctx.dim() == 3:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1, -1)

        ctx = ctx.permute(1, 0, 2, 3)
        ctx = ctx.contiguous().view(self.N * self.n_cls, self.n_ctx, ctx.shape[3])

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.n_cls = len(classnames)
        self.clip_model=clip_model
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.device = torch.device("cuda:0")
        self.device1 = torch.device("cuda")
        self.N = cfg.TRAINER.APP.N
        self.dataset = cfg.DATASET.NAME
        self.use_uniform = True
        self.eps = 0.1
        self.alpha=cfg.alpha
        self.max_iter = 100
        self.meta_feature=None
        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        self.zero_prompt = [temp.format(c.replace("_", " ")) for c in classnames]
        self.zero_prompt  = torch.cat([clip.tokenize(p) for p in self.zero_prompt ]).to(self.device)
    def pretrained_test(self, image):

        b = image.shape[0]
        image_features = self.image_encoder(image.type(self.dtype))
        image_feature_pool = image_features[0]
        image_features = image_features[1:]
        M = image_features.shape[0]
        self.d = image_features.shape[-1]

        tokenized_prompts = self.tokenized_prompts.view(self.N,self.n_cls,-1)

        image_feature_pool = F.normalize(image_feature_pool, dim=1)

        meta_feature = self.prompt_learner.meta_net(image_feature_pool)

        meta_feature_cat = torch.cat([meta_feature.unsqueeze(1)] * self.prompt_learner.n_ctx, dim=1)

        prompt=meta_feature_cat



        meta_feature=meta_feature_cat


        if meta_feature_cat.dim() == 3:
            meta_feature_cat = meta_feature_cat.unsqueeze(0).expand(self.n_cls, -1, -1, -1)

        meta_feature_cat = meta_feature_cat.permute(1, 0, 2, 3)
        meta_feature_cat = meta_feature_cat.contiguous().view(self.n_cls, meta_feature_cat.shape[0],
                                                              self.prompt_learner.n_ctx, meta_feature_cat.shape[3])

        prefix = self.prompt_learner.token_prefix
        suffix = self.prompt_learner.token_suffix
        text_features=[]
        for i in range(meta_feature_cat.shape[1]):

            meta_prompts = torch.cat(
                [
                    torch.mean(prefix.view(self.N, self.n_cls, -1, 512), dim=0),  # (n_cls, 1, dim)
                    meta_feature_cat[:,i],  # (n_cls, n_ctx, dim)
                    torch.mean(suffix.view(self.N, self.n_cls, -1, 512), dim=0),  # (n_cls, *, dim)
                ],
                dim=1,
            )

            meta_text_feature=F.normalize(self.text_encoder(meta_prompts, tokenized_prompts[0]),dim=1)
            text_features.append(meta_text_feature)
        logit_scale = self.logit_scale.exp()

        logits = []
        for i in range(len(text_features)):
            logits.append(logit_scale*torch.einsum('d,cd->c', image_feature_pool[i], text_features[i]))
        logits=torch.stack(logits)
        return logits,meta_prompts








    def pretrained(self, image):

        b = image.shape[0]
        image_features = self.image_encoder(image.type(self.dtype))
        image_feature_pool = image_features[0]
        image_features = image_features[1:]
        M = image_features.shape[0]
        self.d = image_features.shape[-1]

        tokenized_prompts = self.tokenized_prompts.view(self.N,self.n_cls,-1)

        image_feature_pool = F.normalize(image_feature_pool, dim=1)

        meta_feature = self.prompt_learner.meta_net(image_feature_pool)

        meta_feature_cat = torch.cat([meta_feature.unsqueeze(1)] * self.prompt_learner.n_ctx, dim=1)

        prompt=meta_feature_cat



        meta_feature=meta_feature_cat

        if meta_feature_cat.dim() == 3:
            meta_feature_cat = meta_feature_cat.unsqueeze(0).expand(self.n_cls, -1, -1, -1)

        meta_feature_cat = meta_feature_cat.permute(1, 0, 2, 3)
        meta_feature_cat = meta_feature_cat.contiguous().view(self.n_cls, meta_feature_cat.shape[0],
                                                              self.prompt_learner.n_ctx, meta_feature_cat.shape[3])

        prefix = self.prompt_learner.token_prefix
        suffix = self.prompt_learner.token_suffix
        meta_prompts = torch.cat(
            [
                torch.mean(prefix.view(self.N, self.n_cls, -1, 512), dim=0),  # (n_cls, 1, dim)
                torch.mean(meta_feature_cat,dim=1),  # (n_cls, n_ctx, dim)
                torch.mean(suffix.view(self.N, self.n_cls, -1, 512), dim=0),  # (n_cls, *, dim)
            ],
            dim=1,
        )

        meta_text_feature = F.normalize(self.text_encoder(meta_prompts, tokenized_prompts[0]), dim=-1)


        logit_scale = self.logit_scale.exp()
        logits=logit_scale * torch.einsum('bd,cd->bc', image_feature_pool, meta_text_feature)

        return logits, meta_prompts
    




    # def pretrained(self, image,label):
    #
    #     b = image.shape[0]
    #     image_features = self.image_encoder(image.type(self.dtype))
    #     image_feature_pool = image_features[0]
    #     image_features = image_features[1:]
    #     M = image_features.shape[0]
    #     self.d = image_features.shape[-1]
    #
    #     tokenized_prompts = self.tokenized_prompts.view(self.N,self.n_cls,-1)
    #
    #     image_feature_pool = F.normalize(image_feature_pool, dim=1)
    #
    #     meta_feature = self.prompt_learner.meta_net(image_feature_pool)
    #
    #     meta_feature_cat = torch.cat([meta_feature.unsqueeze(1)] * self.prompt_learner.n_ctx, dim=1)
    #
    #     prompt=meta_feature_cat
    #
    #
    #
    #     meta_feature=meta_feature_cat
    #
    #     if meta_feature_cat.dim() == 3:
    #         meta_feature_cat = meta_feature_cat.unsqueeze(0).expand(self.n_cls, -1, -1, -1)
    #
    #     meta_feature_cat = meta_feature_cat.permute(1, 0, 2, 3)
    #     meta_feature_cat = meta_feature_cat.contiguous().view(self.n_cls, meta_feature_cat.shape[0],
    #                                                           self.prompt_learner.n_ctx, meta_feature_cat.shape[3])
    #
    #     prefix = self.prompt_learner.token_prefix
    #     suffix = self.prompt_learner.token_suffix
    #     text_features = []
    #     for i in range(meta_feature_cat.shape[1]):
    #         meta_prompts = torch.cat(
    #             [
    #                 torch.mean(prefix.view(self.N, self.n_cls, -1, 512), dim=0)[label[i]],  # (n_cls, 1, dim)
    #                 meta_feature_cat[label[i], i],  # (n_cls, n_ctx, dim)
    #                 torch.mean(suffix.view(self.N, self.n_cls, -1, 512), dim=0)[label[i]],  # (n_cls, *, dim)
    #             ],
    #             dim=0,
    #         )
    #         meta_prompts=meta_prompts.unsqueeze(0)
    #         token=tokenized_prompts[0, label[i]].unsqueeze(0)
    #         meta_text_feature = F.normalize(self.text_encoder(meta_prompts,token), dim=-1)
    #         text_features.append(meta_text_feature)
    #     logit_scale = self.logit_scale.exp()
    #
    #     logits = []
    #     for i in range(len(text_features)):
    #         logits.append(logit_scale*torch.einsum('d,cd->c', image_feature_pool[i], text_features[i]))
    #     logits = torch.stack(logits)
    #     return logits, meta_prompts







    def forward(self, image):

        b = image.shape[0]
        image_features = self.image_encoder(image.type(self.dtype))
        image_feature_pool = image_features[0]
        image_features = image_features[1:]
        M = image_features.shape[0]
        self.d = image_features.shape[-1]
        image_feature_pool = F.normalize(image_feature_pool, dim=1)
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features.contiguous().view(self.N, self.n_cls, self.d)
        text_feature_pool = text_features

        text_feature_pool = F.normalize(text_feature_pool, dim=2)
#         prompts_copy=copy.deepcopy(prompts)
#         logit_prior, prompt_prior = self.pretrained(image)

        # with torch.no_grad():
        #     logit_prior, prompt_prior = self.pretrained_test(image)
        #


        meta_feature = self.prompt_learner.meta_net(image_feature_pool).detach()

        prompts = self.prompt_learner.ctx

        meta_mean=torch.mean(meta_feature,dim=0).detach()

        prompts=prompts.reshape(self.N,-1,prompts.shape[-1])
        regular=0

        # text_zero=self.clip_model.encode_text(self.zero_prompt)
        # text_zero = F.normalize(text_zero, dim=-1)

        for i in range(self.N):
           regular+=torch.sum(((torch.mean(prompts[i],dim=0)-meta_mean)**2))

        logit_scale = self.logit_scale.exp()
#         logits = logit_scale * torch.einsum('bd,cd->bc', image_feature_pool, torch.mean(text_features_test,dim=0))
#         logits = logit_scale * torch.einsum('bd,cd->bc', image_feature_pool, torch.mean(text_feature_pool,dim=0))
        # logits_zero=logit_scale* torch.einsum('bd,cd->bc', image_feature_pool, text_zero)
        logits2= logit_scale * torch.einsum('bd,ncd->nbc', image_feature_pool, text_feature_pool)
        # logits=(logit_prior)*0.3+logits*0.7


        return logits2,logits2,prompts, regular

    def forward_test(self, image):

        b = image.shape[0]
        image_features = self.image_encoder(image.type(self.dtype))
        image_feature_pool = image_features[0]
        image_features = image_features[1:]
        M = image_features.shape[0]
        self.d = image_features.shape[-1]
        image_feature_pool = F.normalize(image_feature_pool, dim=1)
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features.contiguous().view(self.N, self.n_cls, self.d)
        text_feature_pool = text_features

        text_feature_pool = F.normalize(text_feature_pool, dim=2)
        #         prompts_copy=copy.deepcopy(prompts)
        #         logit_prior, prompt_prior = self.pretrained(image)

        logit_prior, prompt_prior = self.pretrained_test(image)
        #

        meta_feature = self.prompt_learner.meta_net(image_feature_pool).detach()

        prompts = self.prompt_learner.ctx

        meta_mean = torch.mean(meta_feature, dim=0).detach()

        prompts = prompts.reshape(self.N, -1, prompts.shape[-1])
        regular = 0


        logit_scale = self.logit_scale.exp()
        logits = logit_scale * torch.einsum('bd,cd->bc', image_feature_pool, torch.mean(text_feature_pool,dim=0))
        # logits_zero=logit_scale* torch.einsum('bd,cd->bc', image_feature_pool, text_zero)
        logits2 = logit_scale * torch.einsum('bd,ncd->nbc', image_feature_pool, text_feature_pool)
        logits=(logit_prior)*(1-self.alpha)+logits*self.alpha

        return logits, logits2, prompts, regular


@TRAINER_REGISTRY.register()
class APP(TrainerX):
    """
    It is based on CoOp.
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.APP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.APP.PREC == "fp32" or cfg.TRAINER.APP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)
        self.model.to(self.device)

        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.APP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        # device_count = torch.cuda.device_count()
        # if device_count > 1:
        #     print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
        #     self.model = nn.DataParallel(self.model)

    def pretrain_batch(self, batch):
        image, label = self.parse_batch_train(batch)

        prec = self.cfg.TRAINER.APP.PREC
        if prec == "amp":
            with autocast():
                output, prompt = self.model(image)
                loss = F.cross_entropy(output, label)

            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output,feature = self.model.pretrained(image)
            # output,feature = self.model.pretrained(image,label)
            # loss=-torch.mean(output)
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)
        loss_summary = {
            "loss": loss.item(),
        }
        #
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary,feature

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        prec = self.cfg.TRAINER.APP.PREC
        if prec == "amp":
            with autocast():
                output, prompt = self.model(image)
                loss = F.cross_entropy(output, label)

            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output,output1,prompts, regular = self.model(image)
            loss=0
            for i in range(output1.shape[0]):
                loss += F.cross_entropy(output1[i], label)
            loss += (regular*(self.cfg.reg))

            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(torch.mean(output1,dim=0), label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

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
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            self._models[name].load_state_dict(state_dict, strict=False)



