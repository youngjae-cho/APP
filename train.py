import argparse
import torch

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer


import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet

import datasets.imagenet_sketch
import datasets.imagenetv2
import datasets.imagenet_a
import datasets.imagenet_r

import trainers.app


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


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
    if args.reg:
        cfg.reg=args.reg
    if args.alpha:
        cfg.alpha=args.alpha


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

    cfg.TRAINER.APP = CN()
    cfg.TRAINER.APP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.APP.CSC = False  # class-specific context
    cfg.TRAINER.APP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.APP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.APP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.APP.N = 4 # the number of prompts

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)
        
    # cfg.OPTIM.LR=cfg.OPTIM.LR*2


    if cfg.DATASET.NUM_SHOTS == 1:
        cfg.OPTIM.MAX_EPOCH=50//5
    elif cfg.DATASET.NUM_SHOTS == 2 or cfg.DATASET.NUM_SHOTS ==4:
        cfg.OPTIM.MAX_EPOCH=100//5
    else:
        cfg.OPTIM.MAX_EPOCH=200//5


    if cfg.DATASET.SUBSAMPLE_CLASSES =="base" or cfg.DATASET.SUBSAMPLE_CLASSES =="new":
        cfg.OPTIM.MAX_EPOCH=100//5

    cfg.reg = cfg.reg/ (cfg.DATASET.NUM_SHOTS)

    if cfg.DATASET.NAME== "ImageNet" or cfg.DATASET.NAME== "ImageNetA" or cfg.DATASET.NAME== "ImageNetR" or cfg.DATASET.NAME== "ImageNetV2" or cfg.DATASET.NAME== "ImageNetSketch":
        cfg.OPTIM.MAX_EPOCH=50//5

    if cfg.DATASET.NAME in ['EuroSAT','OxfordPets', 'Food101']:
        cfg.reg=cfg.reg/(cfg.DATASET.NUM_SHOTS)
  # 32 for small dataset such as Car,Air,Flowers

    if cfg.DATASET.NAME in ['OxfordFlowers','FGVCAircraft','StanfordCars']:
        cfg.DATALOADER.TRAIN_X.BATCH_SIZE=32  #32 for small dataset such as Car,Air,Flowers
        cfg.reg=cfg.reg/(cfg.DATASET.NUM_SHOTS)
    cfg.DATALOADER.TRAIN_X.BATCH_SIZE1=cfg.DATALOADER.TRAIN_X.BATCH_SIZE
    # cfg.DATALOADER.TRAIN_X.BATCH_SIZE=10
    return cfg



def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)
    args.load_epoch=cfg.OPTIM.MAX_EPOCH*5

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))
    trainer = build_trainer(cfg)
    # trainer.load_model(cfg.OUTPUT_DIR, epoch=100)
    # trainer.visualize()
    # trainer.test()
    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    if not args.no_train:
        trainer.pretrain()
        trainer.build_model_pretrained()
        trainer.train()
    # # #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")

    parser.add_argument("--reg", type=float, default=1.0, help="reg")
    parser.add_argument("--alpha", type=float, default=1.0, help="alpha")

    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)
