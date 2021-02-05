import os
from argparse import ArgumentParser

from config import ROOT, REDDIT_COMMENT_RAW_DATA_DIR, REDDIT_SUBMISSION_RAW_DATA_DIR, REDDIT_PROCESSED_DATA_DIR
from utils import ensure_folder


def run_filter(fn, reddit_type):
    cmd_filter = "bash filter_reddit_posts.sh {input} {output} {command}"
    if fn.endswith('.xz'):
        command = 'xzgrep'
    elif fn.endswith('.bz2'):
        command = 'bzgrep'
    elif fn.endswith('.zst'):
        # .1 for exception in 2019-09
        command = 'zstdgrep'
    elif fn.endswith('.gz'):
        command = 'zgrep'
    else:
        raise RuntimeError("ERROR: unknown input format @ {}".format(fn))

    fn_basename = os.path.basename(fn).split('.')[0]
    output = os.path.join(REDDIT_PROCESSED_DATA_DIR, 'reddit_{}_filtered'.format(reddit_type), fn_basename+'.json.gz')
    ensure_folder(output)

    os.system(cmd_filter.format(input=fn, output=output, command=command))
    return output, fn_basename


if __name__ == '__main__':
    parser = ArgumentParser(description='Ecigs Reddit pipeline')
    parser.add_argument('--input', help='Input data file name or the whole path')
    parser.add_argument('--reddit-type', choices=['submission', 'comment'], help='submission or comment')

    args = parser.parse_args()

    REDDIT_RAW_DATA_DIR = REDDIT_COMMENT_RAW_DATA_DIR if args.reddit_type == 'comment' \
        else REDDIT_SUBMISSION_RAW_DATA_DIR

    if not os.path.exists(args.input):
        if not os.path.exists(os.path.join(REDDIT_RAW_DATA_DIR, args.input)):
            raise RuntimeError("ERROR: input not found @ {}".format(args.input))

    cmd_subreddit = "bash run.sh {fn} {root} reddit_{reddit_type} {out_dir} reddit_subreddit"
    cmd_keywords = "bash run.sh {fn} {root} reddit_{reddit_type} {out_dir} reddit_keywords"
    cmd_count_keywords = "python get_keywords_mentions.py --input {fn} --out-dir {out_dir} --platform reddit"

    # use grep-family commands to filter first
    filter_out_fn, fname = run_filter(args.input, args.reddit_type)

    # use subreddit annotator in Falconet
    subreddit_out_dir = os.path.join(REDDIT_PROCESSED_DATA_DIR, 'reddit_{}'.format(args.reddit_type))
    ensure_folder(subreddit_out_dir)

    if not os.path.exists(filter_out_fn):
        raise RuntimeError("ERROR: input for Reddit subreddit filter not found @ {}".format(filter_out_fn))

    os.system(cmd_subreddit.format(fn=filter_out_fn, root=ROOT, reddit_type=args.reddit_type,
                                   out_dir=subreddit_out_dir))

    # use keywords annotator in Falconet
    keywords_input = os.path.join(REDDIT_PROCESSED_DATA_DIR, 'reddit_{}'.format(args.reddit_type),
                                  'output_reddit_subreddit', 'reddit_{}_{}_out.json.gz'.format(args.reddit_type, fname))
    keywords_out_dir = os.path.join(REDDIT_PROCESSED_DATA_DIR, 'reddit_{}'.format(args.reddit_type))

    if not os.path.exists(keywords_input):
        raise RuntimeError("ERROR: input for Reddit keywords filter not found @ {}".format(keywords_input))

    os.system(cmd_keywords.format(fn=keywords_input, root=ROOT, reddit_type=args.reddit_type, out_dir=keywords_out_dir))

    # count keywords mention
    # This is also the input for counting keywords
    keywords_out_fn = os.path.join(REDDIT_PROCESSED_DATA_DIR, 'reddit_{}'.format(args.reddit_type),
                                   'output_reddit_keywords',
                                   'reddit_{type}_reddit_{type}_{fn}_out_out.json.gz'.format(type=args.reddit_type,
                                                                                             fn=fname))
    keywords_count_out_dir = os.path.join(REDDIT_PROCESSED_DATA_DIR, 'reddit_{}'.format(args.reddit_type),
                                          'output_reddit_keywords')

    if not os.path.exists(keywords_out_fn):
        raise RuntimeError("ERROR: input for counting keywords not found @ {}".format(keywords_out_fn))

    os.system(cmd_count_keywords.format(fn=keywords_out_fn, out_dir=keywords_count_out_dir))
