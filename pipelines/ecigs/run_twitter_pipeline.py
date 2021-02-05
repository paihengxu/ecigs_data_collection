import os
from argparse import ArgumentParser

from config import ROOT, TWITTER_RAW_DATA_DIR, TWITTER_PROCESSED_DATA_DIR
from utils import ensure_folder


if __name__ == '__main__':
    parser = ArgumentParser(description='Ecigs Twitter pipeline')
    parser.add_argument('--input', help='Input data file name or the whole path')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        if not os.path.exists(os.path.join(TWITTER_RAW_DATA_DIR, args.input)):
            raise RuntimeError("ERROR: input not found @ {}".format(args.input))

    cmd_keywords = "bash run.sh {fn} {root} twitter {out_dir} twitter_keywords"
    cmd_relevance = "bash run.sh {fn} {root} twitter {out_dir} twitter_relevance"
    cmd_count_keywords = "python get_keywords_mentions.py --input {fn} --out-dir {out_dir} --platform twitter"

    # keywords annotator
    fname = os.path.basename(args.input).split('.')[0]
    keywords_out_dir = TWITTER_PROCESSED_DATA_DIR

    ensure_folder(keywords_out_dir)

    os.system(cmd_keywords.format(fn=args.input, root=ROOT, out_dir=keywords_out_dir))

    # relevance annotator
    relevance_input = os.path.join(TWITTER_PROCESSED_DATA_DIR, 'output_twitter_keywords',
                                   'twitter_{}_out.json.gz'.format(fname))
    relevance_out_dir = TWITTER_PROCESSED_DATA_DIR

    if not os.path.exists(relevance_input):
        raise RuntimeError("ERROR: input for Twitter relevance classifier not found @ {}".format(relevance_input))

    os.system(cmd_relevance.format(fn=relevance_input, root=ROOT, out_dir=relevance_out_dir))

    # count keywords mention
    # This is also the input for counting keywords
    relevance_out_fn = os.path.join(TWITTER_PROCESSED_DATA_DIR, 'output_twitter_relevance',
                                    'twitter_twitter_{}_out_out.json.gz'.format(fname))
    keywords_count_out_dir = os.path.join(TWITTER_PROCESSED_DATA_DIR, 'output_twitter_relevance')

    if not os.path.exists(relevance_out_fn):
        raise RuntimeError("ERROR: input for counting keywords not found @ {}".format(relevance_out_fn))

    os.system(cmd_count_keywords.format(fn=relevance_out_fn, out_dir=keywords_count_out_dir))
