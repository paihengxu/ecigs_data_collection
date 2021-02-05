from datetime import datetime, timedelta
import pytz

from falconet.counter.data_exporter import DataExporter
from falconet import settings


def create_commandline_options():
    from argparse import ArgumentParser

    # parse command line arguments
    op = ArgumentParser()
    op.add_argument("--input-path", required=True,
                    dest='input_path',
                    help='The path to the count files.')
    op.add_argument("--output-file", required=True, 
                    dest='output_file',
                    help='The file to store the database updates.')
    op.add_argument('--timezone', 
                    dest='timezone', default='US/Eastern',
                    help='The timezone to use for setting dates.')
    op.add_argument('--gap-file', required=True,
                    dest='gap_file',
                    help='A file to write gaps in the data.')
    op.add_argument('--use-history',
                    dest='use_history', type=int,
                    help='Only output stats from the previous N days; if omitted, output all stats in the input files.')
    op.add_argument('--output-format', 
                    dest='output_format', choices=['json', 'postgresql'], default='postgresql',
                    help='The format of the output file. It can be either json or sql.')
    return op.parse_args()
    

def main():
    args = create_commandline_options()
    
    timezone = pytz.timezone(args.timezone)
    current = datetime.now(timezone)
    
    load_after_date = None
    output_after_date = None
    if args.use_history is not None:
        history = args.use_history
        output_after_date = current - timedelta(days=1*history)
        # NOTE: this feature is disabled as we no longer assume the file name has date information.
        # load_after_date = current - timedelta(days=7*history)
    
    if args.output_format == 'postgresql':
        output_format = settings.OUTPUT_FORMAT_POSTGRESQL
    elif args.output_format == 'json':
        output_format = settings.OUTPUT_FORMAT_JSON
    else:
        raise ValueError("unrecognized output format:%s" % args.output_format)
        
    exporter = DataExporter(timezone)
    exporter.run(args.input_path, args.output_file, output_format, args.gap_file, 
                 load_after_date, output_after_date)

if __name__ == '__main__':
    main()
