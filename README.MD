# E-cigarettes Data Collection for Twitter and Reddit
This readme contains the steps for collecting E-cigarettes related posts from Twitter and Reddit.
The pipelines are based on [Falconet](README.Falconet.MD).

![Twitter](figures/twitter.png) ![Reddit](figures/reddit.png)

## Installing

 1.  Create and activate virtual environment
 2.  `pip install -r requirements.txt`
 3.  Copy `python/falconet/settings/local.py.template` to `python/falconet/settings/local.py` and fill in variables.
 4.  Copy `pipelines/ecigs/config.py.template` to `pipelines/ecigs/config.py` and fill in variables.
 5.  Follow the instructions below and run the pipelines.


## Instructions
### Run pipelines
The instructions to process the data are in `pipelines/ecigs`.

- Twitter pipeline contains keywords filtering, relevance classifier, and geolocation inference, which is covered in `run_twitter_pipeline.py`.
- Reddit pipeline contains subreddit filtering, keyword filtering, and geolocation inference.
After running `run_reddit_pipeline.py` for subreddit and keyword filtering, follow the instructions from [SMGEO](https://github.com/kharrigian/smgeo)
and run `run_reddit_geo.py`.

The details of running the scripts are in [Twitter pipeline](pipelines/ecigs/README.MD#Twitter) and [Reddit pipeline](pipelines/ecigs/README.MD#Reddit) respectively.

### Processed data directory
The processed data will be in the path specified in `pipelines/ecigs/config.py`. 
The details of how the outputs are organized are in [Twitter processed data directory](pipelines/ecigs/README.MD#twitter-processed-data-directory) and
[Reddit processed data directory](pipelines/ecigs/README.MD#reddit-processed-data-directory).

For each processed message, we keep the full json record with an additional `annotations` field which contains annotated information through the pipelines.
To keep track of the annotations, the pipeline also output csv files with raw counts of posts and predefined aggregations.
Example [here](README.Falconet.MD#system-overview).

### Aggregation scripts
To help extract information from the output csv files, we provide the instructions for [Twitter](pipelines/ecigs/README.MD#twitter-aggregation-scripts) and
[Reddit](pipelines/ecigs/README.MD#reddit-aggregation-scripts).
These scripts aggregate all csv files in each directory, so it also works when the job is split.


### Pipeline configuration details
We try to be flexible with the pipeline configuration. 
You could follow the instructions [here](README.Falconet.MD#creating-a-new-pipeline) to modify `pipelines/ecigs/*_pipeline.json` or 
create your own pipeline by creating a subdirectory under `pipelines/`.

For Twitter, we provide details on [keyword lists](pipelines/ecigs/README.MD#twitter-keywords-lists),
[tobacco relevance classifiers](pipelines/ecigs/README.MD#different-relevance-classifier-threshold) and 
[how to train the classifiers](pipelines/ecigs/README.MD#training-relevance-classifier).

For Reddit, we provide details on [keyword lists](pipelines/ecigs/README.MD#reddit-keywords-list), 
[relevant Subreddits](pipelines/ecigs/README.MD#subreddits-list) and 
[Reddit raw data retrieval](pipelines/ecigs/README.MD#retrieving-reddit-raw-data).


## Contact
