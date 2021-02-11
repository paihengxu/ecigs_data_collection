#!/bin/bash
ROOT=${2}
PIPELINE=${ROOT}/pipelines/ecigs/${5}_pipeline.json
MESSAGE_TYPE=${3}
OUT_DIR=${4}
OUTPUT=${OUT_DIR}/output_${5}
cd ${ROOT}/python

INPUT=${1}

SOURCE=${MESSAGE_TYPE}
python -m falconet.cli.run $INPUT --pipeline-conf $PIPELINE --output-folder $OUTPUT --output-prefix ${MESSAGE_TYPE} --message-type $SOURCE # --max-messages 1000