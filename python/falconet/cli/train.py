"""
Main method for training classifiers.

Adrian Benton
1/11/2017
"""

import falconet.settings

from falconet.classifiers.train import Tee
from falconet.classifiers.train import initHome, downloadData, downloadEmbeddings
from falconet.classifiers.train import loadData, trainModel
from falconet.classifiers.classifier import TextTokenClassifier

import argparse, datetime, os, sys, time

def main(args):
    """ Called from CLI. """
    initHome(args.home)

    from falconet.classifiers.train import LOG_DIR, DATA_DIR, RSC_DIR, MODEL_DIR

    # reroute output to log file
    if args.outpath:
        sys.stdout = Tee(os.path.join(LOG_DIR, args.outpath))

    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

    if args.downloadall:
        print('Downloading all datasets from bitbucket')
        downloadData(args.bbkey, args.bbsecret, None)
    if not os.path.exists(os.path.join(DATA_DIR, args.inpath)):
        print('Cannot find data at: %s' % (os.path.join(DATA_DIR, args.inpath)))
        print('Falconet requires data to train this model. Falconet will now attempt to download this data from '
              'Bitbucket. Alternatively, you can place the data in the above directory. ')
        downloadData(args.bbkey, args.bbsecret, args.inpath)

    # Download GloVe embeddings
    embeddingPath = os.path.join(RSC_DIR, 'glove.twitter.27B.%dd.txt' % (50))
    if not os.path.exists(embeddingPath):
        if not os.path.exists(RSC_DIR):
            os.mkdir(RSC_DIR)
        downloadEmbeddings()

    print('Training models at %s' % st)

    trainDocs, testDocs, trainLabels, testLabels, folds, tuningFolds, depVars, alphabets = loadData(args.inpath,
                                                                                                    args.dependent,
                                                                                                    args.proptest)
    featureSettings = (args.minngram, args.maxngram, args.embeddingdim)
    regSettings = (args.l1, args.l2)
    # from ipdb import set_trace; set_trace()
    if args.outmodel is not None:
        model_dir = args.outmodel + os.path.basename(args.prefix)     
    else: 
        model_dir = os.path.join(MODEL_DIR, args.prefix)
    # Train models to predict each outcome
    for depVarIdx, depVar in enumerate(depVars):
        DATA = ((trainDocs, [labels[depVarIdx] for labels in trainLabels]),  # train
                (testDocs, [labels[depVarIdx] for labels in testLabels]))  # test
        print('Training to predict "%s"' % (depVar))
        
        trainModel(DATA, folds, tuningFolds, alphabets[depVar], depVar,
                   model_dir, args.beta, args.maxpropdropped,
                   args.numproc, args.inpath, featureSettings, regSettings, args.proptest)


def run_main():
    parser = argparse.ArgumentParser(description='Classifier training harness.')
    parser.add_argument('--inpath', '-i', metavar='IN_PATH',
                        type=str, default=None, required=True,
                        help='path to input data sitting in bitbucket repo')
    parser.add_argument('--outpath', '-o', metavar='LOG_PATH',
                        type=str, default=None,
                        help='path to output log file under LOG_DIR (default writes to stdout)')
    parser.add_argument('--home', metavar='HOME_DIR',
                        required=True,
                        help='directory where data, resources and models will be written')
    parser.add_argument('--outmodel',
                        help='override directory the model will be written')
    parser.add_argument('--prefix', required=True, metavar='PREFIX',
                        help='prefix to model/log file path')

    parser.add_argument('--dependent', '-d', metavar='DEPENDENT_VAR',
                        default=[], nargs='+',
                        help='which dependent variables to build classifiers for.  '
                             'If none are set, this is inferred from training data.')

    grp = parser.add_argument_group(title='model_selection',
                                    description='Determines model selection ' +
                                                '(e.g., how much to favor precision vs. recall, '
                                                'avoid low confidence examples)')
    grp.add_argument('--beta', '-b',
                     type=float, default=1.0,
                     help='weighting of f-score between precision and recall')
    grp.add_argument('--maxpropdropped',
                     type=float, default=0.5,
                     help="what proportion of low confidence examples we allow to be dropped")

    grp = parser.add_argument_group(title='training',
                                    description='Features to extract, regularization constants.  ' +
                                                'Defaults to sweeping over feature sets/regularization if not set.')
    # Training harness now sweeps over feature sets to pick best-performing model, if these are set avoids this sweep
    grp.add_argument('--minngram', metavar='MIN_NGRAM_ORDER',
                     default=None, type=int,
                     help='minimum n-gram order to extract, '
                          'if either  minngram and maxngram are set to -1, extracts no n-gram features')
    grp.add_argument('--maxngram', metavar='MAX_NGRAM_ORDER',
                     default=None, type=int,
                     help='maximum n-gram order to extract')
    grp.add_argument('--embeddingdim', metavar='EMBEDDING_DIM',
                     default=None, type=int,
                     help='which dimensionality of Twitter-trained GloVE embeddings '
                          'to train with -- downloads them if not available')
    grp.add_argument('--l1', metavar='L1_REGULARIZATION',
                     default=None, type=float,
                     help='L1 regularization of trained model; if not set, finds best-performing setting')
    grp.add_argument('--l2', metavar='L2_REGULARIZATION',
                     default=None, type=float,
                     help='L2 regularization when training model')
    grp.add_argument('--proptest', default=None, type=float,
                     help='proportion of data to be heldout for test, if fold is not set in the example')

    grp = parser.add_argument_group(title='bitbucket', description='OAuth creds to access data in Bitbucket repo')
    grp.add_argument('--bbkey', help = 'Bitbucket OAuth key with READ permissiong on "repositories"')
    grp.add_argument('--bbsecret', help='Bitbucket OAuth secret')
    grp.add_argument('--downloadall',
                     default=False,
                     action='store_true',
                     help='if set, downloads all training data from Bitbucket repository')

    parser.add_argument('--numproc', metavar='NUM_PROC',
                        default=1, type=int, help='number of processors to train with')

    current_args = parser.parse_args()

    main(current_args)


if __name__ == '__main__':
    run_main()