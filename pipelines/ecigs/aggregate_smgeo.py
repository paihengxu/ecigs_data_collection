from collections import defaultdict
import pandas as pd
import glob
import sys
import os

from config import SMGEO_OUT_FN

geo_dir = os.path.dirname(SMGEO_OUT_FN)

fn_patterns = [os.path.join(geo_dir, '*.csv')]  # Fill in the output csv files or file patterns from smgeo.
fn_list = []
for fn_pattern in fn_patterns:
    fn_list += glob.glob(fn_pattern)
assert len(fn_list) is not 0, "Couldn't find reddit geolocation csv files to aggregate."

state_count = defaultdict(int)
country_count = defaultdict(int)
for fn in sorted(fn_list):
    df = pd.read_csv(fn)
    us_df = df[df['country_argmax'] == 'US']
    state_values = set(us_df['state_argmax'].values)

    for state in state_values:
        tmp_df = us_df[us_df['state_argmax'] == state]
        if state == 'Washington, D.C.':
            state = 'District of Columbia'
        state_count[state] += len(tmp_df)

    country_values = set(df['country_argmax'].values)
    for country in country_values:
        tmp_df = df[df['country_argmax'] == country]
        country_count[country] += len(tmp_df)

with open('reddit_geo_state_count.csv', 'w') as outf:
    outf.write("{},{}\n".format('state', 'num_user'))
    for k, v in state_count.items():
        outf.write("{},{}\n".format(k, v))

with open('reddit_geo_country_count.csv', 'w') as outf:
    outf.write("{},{}\n".format('country', 'num_user'))
    for k, v in country_count.items():
        outf.write("{},{}\n".format(k, v))





