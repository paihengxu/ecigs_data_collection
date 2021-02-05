#!/bin/sh
# Sweeps over training parameters and picks model with best f-score

ROOT=/export/c10/pxu/falconet/ # The ROOT directory in config.py
#DATA_NAME=${1}
DATA_NAME=${ROOT}/pipelines/ecigs/smokeng/smokeng_tobacco_relevance
#MAX_PROP_DROPPED=${2} # maximum proportion of examples to drop
MAX_PROP_DROPPED=0.25
#DEP_VARS=${3}  # labels to train for
DEP_VARS='relevance'
#BETA=${4}      # how to weight precision and recall in model selection
BETA=0.5
#DIM=${5}
DIM=50   # embedding dimension
NUM_PROC=1
PROP_TEST=0.4

# Need to run me from the scripts directory
cd ${ROOT}/python

python -m falconet.cli.train --inpath ${DATA_NAME}.json.gz --outpath ${DATA_NAME}_dropped_${MAX_PROP_DROPPED}_beta_${BETA}_embed_${DIM}.log --proptest ${PROP_TEST} --beta ${BETA} --maxpropdropped ${MAX_PROP_DROPPED} --dependent ${DEP_VARS} --numproc ${NUM_PROC} --prefix ${DATA_NAME} --home ${ROOT} --embeddingdim ${DIM} #--balance

