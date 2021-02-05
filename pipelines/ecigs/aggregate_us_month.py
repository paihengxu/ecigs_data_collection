import os
import glob
from collections import defaultdict

from config import REDDIT_PROCESSED_DATA_DIR
from aggregate_counter import process_dic


def aggregate_counts(in_dir):
    result = defaultdict(int)
    for fn in glob.glob(os.path.join(in_dir, '*.csv')):
        for idx, line in enumerate(open(fn, 'r')):
            if idx == 0:
                continue
            month, num = line.strip().split(',')
            result[month] += int(num)

    return result


for reddit_type in ['comment', 'submission']:
    # in us
    us_count_dir = os.path.join(REDDIT_PROCESSED_DATA_DIR, 'reddit_{}'.format(reddit_type),
                                'output_reddit_keywords', 'us_count_month')
    us_result = aggregate_counts(us_count_dir)
    process_dic(us_result, sort_pos=0, reverse=False, output='reddit_{}_keywords_us_month'.format(reddit_type),
                col_name='month')

    # with geo info
    geo_count_dir = os.path.join(REDDIT_PROCESSED_DATA_DIR, 'reddit_{}'.format(reddit_type),
                                 'output_reddit_keywords', 'geo_count_month')
    geo_result = aggregate_counts(geo_count_dir)
    process_dic(geo_result, sort_pos=0, reverse=False, output='reddit_{}_keywords_geo_month'.format(reddit_type),
                col_name='month')
