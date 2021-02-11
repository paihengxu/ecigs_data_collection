## Handles training classifiers.

For a sample call used to train tobacco relevance classifier, see [tobacco_relevance_trainer.sh](../../../pipelines/ecigs/smokeng/tobacco_relevance_trainer.sh):

`python -m falconet.cli.train --inpath ${DATA_NAME}.json.gz --outpath ${DATA_NAME}_dropped${MAX_PROP_DROPPED}.log --proptest ${PROP_TEST} --beta ${BETA} --maxpropdropped ${MAX_PROP_DROPPED} --dependent ${DEP_VARS} --numproc ${NUM_PROC} --prefix ${DATA_NAME} --home ${HOME}`

To write performance of each model to table, run [collectModelPerf.py](../../../python/scripts/collectModelPerformance.py).

## Model selection

How models are tuned and evaluated depends on the value of the "fold" field in the input file.
There are three main ways of setting the folds (and how models are selected and evaluated):

- fold set to "train" or "test" -- models are selected by 4-fold CV on training set, and test set
  is used for evaluation
- fold set to train, dev, or test -- fits models on train, selects model on dev, evaluates on test
- fold set to integers -- highest index is the test set, where CV validation on all other folds
  is used for model selection

Any example without a fold field gets placed into train or test set randomly, with the
proportion in test set by the --proptest command line flag.

