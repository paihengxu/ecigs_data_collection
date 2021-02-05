import glob
import os
from collections import defaultdict

from config import TWITTER_PROCESSED_DATA_DIR

us = ["Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware",
      "District of Columbia", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas",
      "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri",
      "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina",
      "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota",
      "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"]


def collect_counts(fn_pattern, task):
    assert task in ['keywords', 'relevance'], task
    fn_list = glob.glob(fn_pattern)

    country_count = defaultdict(int)
    state_count = defaultdict(int)

    label_count = defaultdict(int)
    label_month_count = defaultdict(int)
    us_label_month_count = defaultdict(int)

    for fn in fn_list:
        # print(fn)
        for idx, line in enumerate(open(fn, 'r')):
            if idx == 0:
                continue
            if task == 'relevance':
                timestamp, state, label, count = line.strip().split(',')

                label_count[label] += int(count)
                if label != '1':
                    continue
                # only keep relevant labels
                label_month_count[timestamp] += int(count)
            else:
                timestamp, state, count = line.strip().split(',')
                label_month_count[timestamp] += int(count)

            if state in us:
                state_count[state] += int(count)
                country_count['us'] += int(count)
                us_label_month_count[timestamp] += int(count)
            else:
                if state == 'UNK':
                    country_count['UNK'] += int(count)
                else:
                    country_count['non-us'] += int(count)

    return country_count, state_count, label_count, label_month_count, us_label_month_count


def write_to_csv(dic, file_name, col_name):
    assert col_name in ['month', 'location', 'label'], col_name
    with open(file_name, 'w') as outf:
        outf.write("{},num_tweets\n".format(col_name))
        for k, v in dic.items():
            outf.write("{},{}\n".format(k, v))


if __name__ == '__main__':
    fn_pattern = os.path.join(TWITTER_PROCESSED_DATA_DIR, 'output_twitter_keywords', '*.csv')
    country_count, state_count, _, month_count, _ = collect_counts(fn_pattern, task='keywords')
    write_to_csv(country_count, 'twitter_keywords_country.csv', col_name='location')
    write_to_csv(state_count, 'twitter_keywords_state.csv', col_name='location')
    write_to_csv(month_count, 'twitter_keywords_month.csv', col_name='month')

    print("keywords filtered tweets from country", sum(country_count.values()))
    print("keywords filtered tweets from month", sum(month_count.values()))
    print("keywords filtered tweets in US from state", sum(state_count.values()))
    print("keywords filtered tweets in US from country", country_count['us'])

    fn_pattern = os.path.join(TWITTER_PROCESSED_DATA_DIR, 'output_twitter_relevance', '*.csv')
    country_count, state_count, label_count, label_month_count, us_label_month_count = collect_counts(fn_pattern, task='relevance')
    write_to_csv(country_count, 'twitter_relevant_country.csv', col_name='location')
    write_to_csv(state_count, 'twitter_relevant_state.csv', col_name='location')
    write_to_csv(label_count, 'twitter_relevant.csv', col_name='label')
    write_to_csv(label_month_count, 'twitter_relevant_month.csv', col_name='month')
    write_to_csv(us_label_month_count, 'twitter_relevant_us_month.csv', col_name='month')
    print("relevant tweets from country", sum(country_count.values()))
    print("relevant tweets from month", sum(label_month_count.values()))
    print("relevant tweets from relevance", label_count['1'])

    print("relevant tweets in US from state", sum(state_count.values()))
    print("relevant tweets in US from country", country_count['us'])
    print("relevant tweets in US from month", sum(us_label_month_count.values()))

    print("keywords filtered tweets from relevance", sum(label_count.values()))

