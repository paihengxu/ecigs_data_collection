#!/bin/bash
ROOT=${2}
PIPELINE=${ROOT}/pipelines/ecigs/${5}_pipeline.json
MESSAGE_TYPE=${3}
OUT_DIR=${4}
OUTPUT=${OUT_DIR}/output_${5}
cd ${ROOT}/python

INPUT=${1}

SOURCE=${MESSAGE_TYPE}
python -m falconet.cli.run $INPUT --pipeline_conf $PIPELINE --output_folder $OUTPUT --output_prefix ${MESSAGE_TYPE} --message_type $SOURCE # --max_messages 1000