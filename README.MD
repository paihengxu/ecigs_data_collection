# E-cigarettes Data Collection for Twitter and Reddit
This readme contains the steps for collecting, filtering, and processing e-cigarettes related posts from Twitter and Reddit.
The pipelines are based on [Falconet](README.Falconet.MD).

![Twitter](figures/twitter.png) ![Reddit](figures/reddit.png)

## Installing

 1.  Create and activate virtual environment
 2.  `pip install -r requirements.txt`
 3.  Copy `python/falconet/settings/local.py.template` to `python/falconet/settings/local.py` and fill in variables.
 4.  Copy `pipelines/ecigs/config.py.template` to `pipelines/ecigs/config.py` and fill in variables.
 5.  Follow the instructions below and run the pipelines.


## Instructions
We provided detailed instructions to process the data in the instructions to process the data are in `pipelines/ecigs`.
We provide an overview of our system and pipelines [here](README.Falconet.MD#system-overview).

### Pipelines
We use seperate pipelines for Twitter and Reddit.

- Twitter pipeline involves keywords filtering, relevance classifier, and geolocation inference, which is covered in `run_twitter_pipeline.py`.
- Reddit pipeline involves subreddit filtering, keyword filtering, and geolocation inference.

We provide details of running the [Twitter pipeline](pipelines/ecigs/README.MD#Twitter) and [Reddit pipeline](pipelines/ecigs/README.MD#Reddit).

### Processed data directory
The processed data will be stored in the path specified in the [configuration file](pipelines/ecigs/config.py). 
The details of how the outputs are organized are in [Twitter processed data directory](pipelines/ecigs/README.MD#twitter-processed-data-directory) and
[Reddit processed data directory](pipelines/ecigs/README.MD#reddit-processed-data-directory).

For each processed message, we keep the full json record with an additional `annotations` field which contains annotated information through the pipelines.
To keep track of the annotations, the pipeline also output csv files with raw counts of posts and predefined aggregations.


### Aggregation scripts
We provide the instructions for extracting information about the data after applying our [Twitter](pipelines/ecigs/README.MD#twitter-aggregation-scripts) and
[Reddit](pipelines/ecigs/README.MD#reddit-aggregation-scripts) pipelines.

### Pipeline configuration details
We try to be flexible with the pipeline configuration. 
Follow the instructions [here](README.Falconet.MD#creating-a-new-pipeline) to either modify `pipelines/ecigs/*_pipeline.json` or 
create your own pipeline by creating a subdirectory under `pipelines/`.

For Twitter, we provide details on [keyword lists](pipelines/ecigs/README.MD#twitter-keywords-lists),
[tobacco relevance classifiers](pipelines/ecigs/README.MD#different-relevance-classifier-threshold) and 
[how to train the classifiers](pipelines/ecigs/README.MD#training-relevance-classifier).

For Reddit, we provide details on [keyword lists](pipelines/ecigs/README.MD#reddit-keywords-list), 
[relevant Subreddits](pipelines/ecigs/README.MD#subreddits-list) and 
[Reddit raw data retrieval](pipelines/ecigs/README.MD#retrieving-reddit-raw-data).


## Contact
Mark Dredze (mdredze at cs.jhu.edu)