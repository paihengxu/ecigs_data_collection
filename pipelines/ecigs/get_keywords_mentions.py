import os
import gzip
import json
from collections import defaultdict
import argparse

from datetime import datetime
from dateutil import parser, tz

from utils import ensure_folder


def get_day_from_date(date_str, time_zone="UTC"):
    """
    Returns the hour string %Y-%m-%d %H:00:00 %z for given input datestring with utc_off_set, default time_zone = UTC
    """

    tzlocal = tz.gettz(time_zone)
    z = (parser.parse(date_str))
    coverted_time_zone = z.astimezone(tzlocal)
    day_str = coverted_time_zone.strftime('%Y-%m-%d')

    return day_str


def get_day_from_timestamp(timestamp):
    return str(datetime.fromtimestamp(int(timestamp)).date())


def main(args):
    out_dir = args.out_dir
    basename = os.path.basename(args.input)
    basename = basename.strip('.json.gz')

    create_time_field_mapping = {
        'reddit': 'created_utc',  # timestamp
        'twitter': 'created_at'   # date_str
    }
    date_transform_func = {
        'reddit': get_day_from_timestamp,
        'twitter': get_day_from_date
    }
    keywords_count = defaultdict(dict)
    for line in gzip.open(args.input, 'r'):
        data = json.loads(line.decode('utf8'))
        kw_list = data['annotations']['keywords']
        for kw in kw_list:
            create_time = data[create_time_field_mapping[args.platform]]
            date_str = date_transform_func[args.platform](create_time)

            if kw not in keywords_count[date_str]:
                keywords_count[date_str][kw] = 1
            else:
                keywords_count[date_str][kw] += 1

    out_fn = os.path.join(out_dir, 'keywords_count')
    out_fn = os.path.join(out_fn, '{}_counts_date_keywords.csv'.format(basename))
    ensure_folder(out_fn)

    with open(out_fn, 'w') as outf:
        outf.write('date,keyword,count\n')
        for date_str, sub_dict in keywords_count.items():
            for kw, count in sub_dict.items():
                outf.write('{},{},{}\n'.format(date_str, kw, count))


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Count annotated keywords')
    arg_parser.add_argument('--input', metavar='input', type=str, required=True,
                            help='annotated .json.gz file to process')
    arg_parser.add_argument('--out-dir', metavar='out_dir', type=str, required=True,
                            help='output directory')
    arg_parser.add_argument('--platform', metavar='platform', choices=['reddit', 'twitter'], required=True,
                            help='reddit or twitter')

    args = arg_parser.parse_args()

    main(args)
    print("Job done!")


