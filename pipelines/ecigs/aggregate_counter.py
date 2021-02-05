from collections import defaultdict
import glob
import os
from datetime import datetime
import sys

from config import TWITTER_PROCESSED_DATA_DIR, REDDIT_PROCESSED_DATA_DIR


def count_stats(fn_patterns):
    fn_list = []
    for fn_pattern in fn_patterns:
        print(fn_pattern)
        fn_list += glob.glob(fn_pattern)
    print("Num of files to process", len(fn_list))

    monthly_data = defaultdict(int)
    interested_data = defaultdict(int)
    for fn in sorted(fn_list):
        # print(fn)
        for idx, line in enumerate(open(fn, 'r')):
            if idx == 0:
                continue
            month, interested_item, count = line.strip().split(',')

            if interested_item == 'UNK':
                continue
            monthly_data[month] += int(count)
            interested_data[interested_item] += int(count)

    return monthly_data, interested_data


def process_dic(dic, sort_pos=0, reverse=False, output=None, col_name=None):
    assert col_name in ['month', 'keywords', 'subreddits']
    dic_list = sorted(dic.items(), key=lambda item: item[sort_pos], reverse=reverse)
    x = []
    y = []
    if output:
        outf = open(output+'.csv', 'w')
        outf.write("{},num\n".format(col_name))
    for idx, num in dic_list:
        x.append(idx)
        y.append(num)
        if output:
            outf.write("{},{}\n".format(idx, num))
    return x, y


def transform_date2month(dic):
    result = defaultdict(int)
    for k, v in dic.items():
        datee = datetime.strptime(k, '%Y-%m-%d')
        month_k = "{}-{}".format(datee.year, str(datee.month).zfill(2))
        result[month_k] += v
    return result


if __name__ == '__main__':
    platform = sys.argv[-1]
    if platform == 'twitter':
        # aggregate keywords count for Twitter
        message_type = 'twitter'
        fn_patterns = [os.path.join(TWITTER_PROCESSED_DATA_DIR, 'output_twitter_relevance', 'keywords_count', '*.csv')]
        date_data, twitter_keywords_data = count_stats(fn_patterns)
        _, _ = process_dic(twitter_keywords_data, sort_pos=0, reverse=True,
                           output='{}_relevant_keywords_counts'.format(message_type), col_name='keywords')

        # transform date dict to month dict
        monthly_data = transform_date2month(date_data)
        _, _ = process_dic(monthly_data, sort_pos=0, reverse=True,
                           output='{}_relevant_keywords_month'.format(message_type), col_name='month')

    elif platform == 'reddit':
        # aggregate month and subreddit data for Reddit, after subreddit and keywords steps
        for message_type in ['comment', 'submission']:
            # subreddit
            fn_patterns = [os.path.join(REDDIT_PROCESSED_DATA_DIR, 'reddit_{}'.format(message_type),
                                        'output_reddit_subreddit', '*.csv')]
            monthly_data, subreddit_data = count_stats(fn_patterns)
            _, _ = process_dic(monthly_data, output='reddit_{}_subreddit_month'.format(message_type), col_name='month')

            _, _ = process_dic(subreddit_data, sort_pos=0, reverse=True,
                               output='reddit_{}_subreddit'.format(message_type), col_name='subreddits')

            # keywords
            fn_patterns = [os.path.join(REDDIT_PROCESSED_DATA_DIR, 'reddit_{}'.format(message_type),
                                        'output_reddit_keywords', '*.csv')]
            monthly_data, subreddit_data = count_stats(fn_patterns)
            _, _ = process_dic(monthly_data, output='reddit_{}_keywords_month'.format(message_type), col_name='month')

            _, _ = process_dic(subreddit_data, sort_pos=0, reverse=True,
                               output='reddit_{}_keywords_subreddit'.format(message_type), col_name='subreddits')

        # aggregate keywords count for Reddit
        fn_patterns = []
        for message_type in ['comment', 'submission']:
            fn_patterns += [os.path.join(REDDIT_PROCESSED_DATA_DIR, 'reddit_{}'.format(message_type),
                                         'output_reddit_keywords', 'keywords_count', '*.csv')]

        message_type = 'reddit'
        date_data, reddit_keywords_data = count_stats(fn_patterns)
        _, _ = process_dic(reddit_keywords_data, sort_pos=0, reverse=True,
                           output='{}_keywords_counts'.format(message_type), col_name='keywords')

        # # transform date dict to month dict
        monthly_data = transform_date2month(date_data)
        _, _ = process_dic(monthly_data, sort_pos=0, reverse=True,
                           output='{}_keywords_counts_month'.format(message_type), col_name='month')
    else:
        raise ValueError("Unknown platform {}. Please enter reddit or twitter".format(platform))
