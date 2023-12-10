




#!/bin/bash


# custom config
DATA=""
TRAINER=APP

DATASET=$1
CFG=rn50  # config file
CTP=end  # class token position (end or middle)
NCTX=16  # number of context tokens
# SHOTS=$5  # number of shots (1, 2, 4, 8, 16)
CSC=False  # class-specific context (False or True)
N=$2  # number of proxy

for SHOTS in 16
do
for SEED in 1 2 3
do
DIR=""
DIR1=""

CUDA_VISIBLE_DEVICES=6 python3.8 train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--output-dir ${DIR} \
--model-dir ${DIR1} \
--eval-only \
--alpha 0.7 \
TRAINER.APP.N_CTX ${NCTX} \
TRAINER.APP.CSC ${CSC} \
TRAINER.APP.CLASS_TOKEN_POSITION ${CTP} \
DATASET.NUM_SHOTS ${SHOTS} \
TRAINER.APP.N ${N} \
DATASET.SUBSAMPLE_CLASSES new

done
done