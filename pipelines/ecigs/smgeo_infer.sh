#!/bin/bash

INPUT=${1}  # txt file with one Reddit author name per line
OUTPUT=${2}  # a csv file with geolocation as output
SMGEO_DIR=${3} # path to SMGEO repo
SMGEO_VENV=${4} # path to SMGEO virtual environment

source ${SMGEO_VENV}/bin/activate

cd ${SMGEO_DIR}

python scripts/model/reddit/infer.py \
	models/reddit/Global_TextSubredditTime/model.joblib \
	${INPUT} \
	${OUTPUT} \
	--start_date 2015-01-01 \
	--comment_limit 200 \
	--known_coordinates \
	--reverse_geocode

deactivate
