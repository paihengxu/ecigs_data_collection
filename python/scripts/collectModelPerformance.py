'''
Writes out performance of each model in a directory to file.  Includes argument dictionary to retrain model
(populate retrainModel.py).

Adrian Benton
2/10/2017
'''

import argparse, os, pickle, re
import pandas as pd


def main(inDir, outPath):
    pathRe = re.compile(
        r'(?P<prefix>.+)_DEP\-(?P<depvar>.+)_DROPPED-(?P<dropped>[0-9\.]+)_NGRAM-\((?P<minngram>[0-9\-]+),(?P<maxngram>[0-9\-]+)\)_BETA-(?P<beta>[0-9\.]+)_EMB-(?P<emb>[0-9\-]+)')
    tblRows = []
    header = ['path', 'prefix', 'label_name', 'max_prop_dropped', 'f1_beta',
              'min_ngram_order', 'max_ngram_order', 'embedding_width',
              'accuracy', 'fscore', 'kept_labels',
              'dropped_labels', 'threshold', 'train_args']
    paths = [os.path.join(inDir, p) for p in os.listdir(inDir) if p.endswith('.pickle')]

    for p in paths:
        f = open(p, 'rb')
        model = pickle.load(f)
        f.close()

        match = pathRe.match(os.path.basename(p))

        prefix = match.group('prefix')
        maxPropDropped = match.group('dropped')
        beta = match.group('beta')
        minNgram = match.group('minngram')
        maxNgram = match.group('maxngram')
        embeddingWidth = match.group('emb')

        row = [p, prefix, model._name, maxPropDropped, beta, minNgram,
               maxNgram, embeddingWidth, model._testAcc, model._testFscore,
               model._testKeptLabelDist, model._testDroppedLabelDist, model._threshold,
               model._trainArgs]

        tblRows.append(row)
        print('.', end='')
    print('')

    tbl = pd.DataFrame(tblRows, columns=header)
    tbl = tbl.sort_values(by=['prefix', 'label_name', 'max_prop_dropped', 'f1_beta'])

    if outPath is not None:
        tbl.to_csv(outPath, sep='\t', index=False)
    else:
        print(tbl.to_csv(outPath, sep='\t', index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classifier training harness.')
    parser.add_argument('--indir', '-i', metavar='IN_DIR',
                        type=str, default=None, required=True,
                        help='directory where models live')
    parser.add_argument('--outpath', '-o', metavar='OUT_PATH',
                        type=str, default=None,
                        help='where to write table, if not set, then writes table to stdout')
    args = parser.parse_args()

    main(args.indir, args.outpath)
