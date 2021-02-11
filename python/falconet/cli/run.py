from argparse import ArgumentParser
import json
import pprint
try:
    from ipdb import set_trace
except ImportError:
    pass

import os

from falconet.io import StreamWriter, TextStreamReader, MinimalWriter
from falconet.pipeline import Pipeline
from falconet.utils import Timer, ensure_folder
from falconet.settings import ID


def command_line_parser():
    parser = ArgumentParser(description='Falconet pipeline')
    parser.add_argument('INPUT',  nargs='+',
                        help='One or multiple paths to the data or more to be processed (file or folder)')
    parser.add_argument('--output-prefix', required=True,
                        help='Prefix of output file name')
    parser.add_argument('--output-folder', required=True,
                        help='Processed data and statistics will be stored in this folder')
    parser.add_argument('--pipeline-conf', type=str, required=True,
                        help='Path to a pipeline config file.')
    parser.add_argument('--message-type', choices=['twitter', 'reddit_comment', 'reddit_submission'],
                        required=True, help='Message type')    
    parser.add_argument('--max-messages', type=int, help='Process this many messages and stop')
    parser.add_argument('--ignore-previous-annotations', action='store_true',
                        help='if specified, do NOT add to an existing dict of annotations')

    args = parser.parse_args()
    return args


def run_pipeline(pipeline_conf, message_type, input_data, output_folder, max_messages=None, use_previous_annotations=True, output_prefix=None):
    input_data = input_data[0].split(',')
    # check if each item in the input_data exists
    for input_data_item in input_data:
        if not os.path.exists(input_data_item):
            raise RuntimeError("ERROR: input not found @ {}".format(input_data_item))
    
    with open(pipeline_conf, "r") as f:
        conf = json.load(f)
        print("> pipeline {}".format(conf["name"]))
        pprint.pprint(conf)
        print("")
    
    # create output folder if needed
    ensure_folder(output_folder)
    
    # get fname
    if len(input_data) == 1:
        # we take the basename
        if not os.path.isdir(input_data[0]):
            fname = os.path.basename(input_data[0]).split('.')[0]
        else:
            fname = os.path.basename(input_data[0].rstrip(os.sep))
    else:
        # get shared parent directory or file name
        fname = os.path.commonprefix(input_data).split(os.sep)[-1]

    # add the prefix
    fname = output_prefix + '_' + fname.rstrip('_')

    output = os.path.join(output_folder, fname + "_out.json.gz")
    counter_output = os.path.join(output_folder, fname + "_counts.csv")
    print("[output @ {}]".format(output))
    print("[counts @ {}]".format(counter_output))

    # create pipeline, reader and writer
    pipeline = Pipeline(conf, counter_output)
    reader = TextStreamReader(
        data_path=input_data, message_type=message_type, 
        max_posts=max_messages, 
        use_previous_annotations=use_previous_annotations
    )
    writer = StreamWriter(output, conf.get("writer-style", ID))
    
    # check if minimial outputs
    mwriter = None
    if 'minimal_output' in conf:
        mwriter = MinimalWriter(
            os.path.join(output_folder, fname + "_minimal.json.gz"),
            conf['minimal_output']
        )

    timer = Timer()
    timer.start()
    # run pipeline on all the data
    for p in reader.stream():
        p_out = pipeline.run(p)
        if 'skip_message' not in p_out.metadata:
            writer.write(p_out)

            # if minimal output is created
            if mwriter:
                mwriter.write(p_out)
    timer.stop()
    print("[Processing time: {}]".format(timer.get_full_time()))
    # close files
    writer.close()
    pipeline.flush_counts()
    if mwriter:
        mwriter.close()


def main():
    args = command_line_parser()

    run_pipeline(
        pipeline_conf=args.pipeline_conf, message_type=args.message_type, 
        input_data=args.INPUT, output_folder=args.output_folder,
        max_messages=args.max_messages, 
        use_previous_annotations=not args.ignore_previous_annotations,
        output_prefix=args.output_prefix
    )


if __name__ == '__main__':
    main()
