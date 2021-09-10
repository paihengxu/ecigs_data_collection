#!/bin/bash
source /export/c10/pxu/venv3.7/bin/activate

python retrieve_reddit_data.py --message-type ${1} --start-date ${2} --end-date ${3} --out-dir ${4}

deactivate