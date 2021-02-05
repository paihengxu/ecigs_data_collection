import pandas as pd
import json
import gzip
import sys
import os
from datetime import datetime
from collections import defaultdict

from utils import ensure_folder
from config import SMGEO_OUT_FN

import argparse

parser = argparse.ArgumentParser(description='Count Reddit data in US')
parser.add_argument('--fn', type=str, required=True)
parser.add_argument('--outfn', type=str, required=True)

args = parser.parse_args()
fn = args.fn
out_fn = args.outfn


def get_month_from_timestamp(timestamp):
    datee = datetime.fromtimestamp(int(timestamp)).date()
    month_k = "{}-{}".format(datee.year, str(datee.month).zfill(2))
    return month_k


geo_df = pd.read_csv(SMGEO_OUT_FN)
if 'author' not in geo_df.columns:
    geo_df.rename(columns={'Unnamed: 0': 'author'}, inplace=True)
geo_df.set_index('author', inplace=True)
result = defaultdict(int)
all_result = defaultdict(int)
all_authors = geo_df.index.values
for line in gzip.open(fn, 'r'):
    post = json.loads(line.decode('utf8'))
    author = post['author']
    month = get_month_from_timestamp(post['created_utc'])
    if author not in all_authors:
        continue
    if geo_df.loc[author, 'country_argmax'] == 'US':
        result[month] += 1
    all_result[month] += 1

in_dir = os.path.dirname(fn)
out_fn_us = os.path.join(in_dir, 'us_count_month', out_fn)
ensure_folder(out_fn_us)

with open(out_fn_us, 'w') as outf:
    outf.write('month,num\n')
    for k, v in result.items():
        outf.write('{},{}\n'.format(k, str(v)))

out_fn_geo = os.path.join(in_dir, 'geo_count_month', out_fn)
ensure_folder(out_fn_geo)

with open(out_fn_geo, 'w') as outf:
    outf.write('month,num\n')
    for k, v in all_result.items():
        outf.write('{},{}\n'.format(k, str(v)))
