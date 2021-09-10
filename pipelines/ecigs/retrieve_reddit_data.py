import os
import time
import json
import gzip
import argparse
from retriever import Reddit

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def read_keywords(fn):
    keywords = set()
    for line in open(fn, 'r'):
        keywords.add(line.strip())
    return keywords


def main(args):
    start = time.time()
    y, m, _ = args.start_date.split('-')

    # init
    # submission_ids_fn = os.path.join(args.out_dir, 'submission_ids', '{}-{}.txt'.format(y, m))
    logging.info("retrieving {} from {} to {}".format(args.message_type, args.start_date, args.end_date))

    wrapper = Reddit()

    out_name_mapping = {'submission': 'RS', 'comment': 'RC'}

    out_fn = os.path.join(args.out_dir, 'reddit_{}'.format(args.message_type),
                          "{}_{}-{}.json.gz".format(out_name_mapping[args.message_type], y, m))

    outf = gzip.open(out_fn, 'w')

    ecig_subreddits_fn = '../../python/falconet/resources/annotators/subreddits/tobacco/e-cigs.txt'
    ecig_subreddits = read_keywords(ecig_subreddits_fn)
    logging.info("{} subreddits to retrieve".format(len(ecig_subreddits)))

    for subreddit in ecig_subreddits:

        logging.info("retrieving from {}".format(subreddit))
        try:
            if args.message_type == 'submission':
                df = wrapper.retrieve_subreddit_submissions(subreddit,
                                                            start_date=args.start_date,
                                                            end_date=args.end_date)
            elif args.message_type == 'comment':
                df = wrapper.search_for_comments(query=None,
                                                 subreddit=subreddit,
                                                 start_date=args.start_date,
                                                 end_date=args.end_date)
            result = df.to_json(orient='records')
        except Exception as err:
            logging.info(err)
            continue
        # print(result)
        parsed = json.loads(result)
        for ele in parsed:
            outf.write("{}\n".format(json.dumps(ele)).encode('utf8'))

    outf.close()

    logging.info("Job done! Process time: {} seconds".format(time.time()-start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retrieve Reddit data')
    parser.add_argument('--message-type', metavar='message_type', choices=['comment', 'submission'], required=True,
                        help='retrieve comment or submission')
    parser.add_argument('--start-date', metavar='start_date', required=True,
                        help='start date, should be in the form of YYYY-MM-DD')
    parser.add_argument('--end-date', metavar='end_date', required=True,
                        help='end date, should be in the form of YYYY-MM-DD')
    parser.add_argument('--out-dir', metavar='out_dir', default='/export/c10/pxu/data/tobacco/reddit/raw_monthly_subreddits',
                        help='path to store retrieved data')
    args = parser.parse_args()

    main(args)