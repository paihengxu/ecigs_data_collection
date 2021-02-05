# E-cigarettes Data Collection for Twitter and Reddit
This readme contains the steps for collecting E-cigarettes related posts from Twitter and Reddit.
The pipelines are based on [Falconet](README.Falconet.MD).

![Twitter](twitter.png) ![Reddit](reddit.png)

## Installing

 1.  Create and activate virtual environment
 2.  `pip install -r requirements.txt`
 3.  Copy `python/falconet/settings/local.py.template` to `python/falconet/settings/local.py` and fill in variables.
 4.  Copy `pipelines/ecigs/config.py.template` to `pipelines/ecigs/config.py` and fill in variables.
 5.  Following the instructions below and run the pipelines.
 
## Twitter
Twitter pipeline contains keywords filtering, relevance classifier, and geolocation inference. 
Run the following command in the `pipelines/ecigs/` directory.

`python run_twitter_pipeline.py --input {input_fn}` 

where `{input_fn}` should be the whole path to the raw data 
or the file name of raw data in the `TWITTER_RAW_DATA_DIR` directory specified in `pipelines/ecigs/config.py`.
 
### Processed data directory
- Keywords filtering and geolocation inference: `{TWITTER_PROCESSED_DATA_DIR}/twitter/output_twitter_keywords/`.
- Relevance classifier: `{TWITTER_PROCESSED_DATA_DIR}/twitter/output_twitter_relevance/`.

For each processed Tweet, we keep the full json record with an additional `annotations` field 
including `keywords`, `location` and `label-relevance` 
(`1` for relevant, `0` for irrelevant, `None` for dropped low confidence examples).

To keep track of the annotations, the pipeline also output csv files with raw counts of posts and predefined aggregations in the above-listed directories.
Additionally, there is another csv file that counts the mentioning times of each keywords per day for Tweets classified as relevant,
in the `{TWITTER_PROCESSED_DATA_DIR}/twitter/output_twitter_relevance/keywords_count/` directory. 

To help extract information from the output csv files, we provide following scripts and commands to run the scripts (in `pipelines/ecigs`).
These scripts aggregate all csv files in each directory, so it also works when the job is split.

- `python aggregate_carmen.py`: generates the following csv files for *counts of posts* in the Tweets collection with keywords or classified as relevant. 
`{step}` could be `keywords` or `relevant`.
    - `twitter_{step}_country.csv`: counts for `us`, `non-us`, and `UNK` (unknown).
    - `twitter_{step}_state.csv`: counts over U.S. states.
    - `twitter_{step}_month.csv`: counts over month.
    - `twitter_relevant_us_month.csv`: counts in U.S. over month.
    - `twitter_relevant.csv`: counts for relevant and irrelevant Tweets.
- `python aggregate_counter.py twitter`: generates the following csv files for *counts of keywords mentioning* in the collection of relevant Tweets.
    - `twitter_relevant_keywords_counts.csv`: counts for each keywords.
    - `twitter_relevant_keywords_month.csv`: counts over month.
 
### Keywords lists
1. [Bigger keyword list](python/falconet/resources/annotators/keywords/ENDS.keywords):
  - extract all tweets that contain the master list of 611 unique [seed brands](python/falconet/resources/annotators/keywords/ecigs_brands.keywords) 
  and 23 [manually defined e-cig keywords](python/falconet/resources/annotators/keywords/ecig.keywords).
2. [Conservative keyword list](python/falconet/resources/annotators/keywords/ENDS_refined.keywords): 
  - extract all tweets that contain the master list of 480 unique seed brands and 23 manually defined e-cig keywords.

The conservative keyword list is used in the pipeline.
 
 
### Different relevance classifier threshold
We provide multiple relevance classifiers with different thresholds for dropping low confidence examples, located in `models`.
The performances for each best model on the test set are as follow.

| Drop percentage | Accuracy | F1 score |
|:---------------:|:--------:|:--------:|
|        0%       |    87%   |   0.85   |
|       25%       |    95%   |   0.94   |
|       50%       |    98%   |   0.98   |

The default model is `models/smokeng_tobacco_relevance_DEP-relevance_DROPPED-0.25_NGRAM-(1,2)_BETA-0.5_EMB-50.pickle`,
as specified in `pipelines/ecigs/twitter_relevance_pipeline.json`. 
To switch to another model, change the corresponding argument of `classifier` in the `pipelines` field 
such as `maxpropdropped`, `embeddingdim` accordingly.
 
### Training relevance classifier
We provide the [Smokeng dataset](pipelines/ecigs/smokeng/smokeng_tobacco_relevance.json.gz) 
and the [script](pipelines/ecigs/smokeng/tobacco_relevance_trainer.sh) to train the relevance classifier.
The details for training classifier with Falconet can be found [here](README.Falconet.MD#training-models). 
 
## Reddit
Reddit pipeline contains subreddit filtering, keywords filtering, and geolocation inference.
To speed up the processing, we use grep-family commands to pre-filter with subreddits, 
which supports compressed files in the following format: `.gz, .xz, .bz2, and .zst`.

For subreddit filtering and keywords filtering, run the following command in the `pipelines/ecigs/` directory.
  
  `python run_reddit_pipeline.py --input {input_fn} --reddit-type {reddit_type}`
  
where

- input_fn: same as Twitter.
- reddit-type: `submission` or `comment`.

### Geolocation inference
First, follow the instructions from [smgeo](https://github.com/kharrigian/smgeo) to set up the virtual environment and models.
The relevant paths should be consistent with the ones you set up in `pipelines/ecigs/config.py`.

Then run `python run_reddit_geo.py` to infer the geolocation information of users who has posted comments or submissions 
with ecigs keywords in relevant subreddits.
A list of Reddit user can be found at `{REDDIT_PROCESSED_DATA_DIR}/reddit_authors.txt`.
The inferred geolocation information is at `{SMGEO_OUT_FN}`.

   
### Processed data directory
Similar to Twitter, there are also csv files with raw counts of posts in the following directories.

- Subreddit filtering: `{REDDIT_PROCESSED_DATA_DIR}/reddit/reddit_{reddit_type}/output_reddit_subreddit/`.
- Keywords filtering: `{REDDIT_PROCESSED_DATA_DIR}/reddit/reddit_{reddit_type}/output_reddit_keywords/`.
    - In its subdirectory `keywords_count`, there is a csv file that counts the mentioning times of each keywords per day.

where `{reddit_type}` is `submission` or `comment`.

To associate users' geolocation information with Reddit posts (submissions or comments), 
run the following command to count the number of posts with geolocation information or in United States.

`python count_reddit_geo.py --fn {input_fn} --outfn {output}`

where

- `{input_fn}`: Keyword filtered data file path, `.gz` files in `{REDDIT_PROCESSED_DATA_DIR}/reddit/reddit_{reddit_type}/output_reddit_keywords/`.
- `{output}`: output file name, in `.csv` format. 

The output files will be in the same directory as `{input_fn}`, and create the following two subdirectories:

- Count of posts with geolocation information will be in `geo_count_month`,
- Counts of posts in U.S. will be in `us_count_month`.

The scripts and commands to run the scripts (in `pipelines/ecigs`) for produce aggregated summaries are as follows:

- `python aggregate_counter.py reddit`:
    - Step 1: subreddit filtering:
        - `reddit_{reddit_type}_subreddit.csv`: counts # of posts in each subreddit.
        - `reddit_{reddit_type}_subreddit_month.csv`: counts # of posts over month.
    - Step 2: keywords filtering:
        - `reddit_{reddit_type}_keywords_subreddit.csv`: counts # of posts in each subreddit.
        - `reddit_{reddit_type}_keywords_month.csv`: counts # of posts over month.
        - `reddit_keywords_counts.csv`: counts # of keyword mentioning times for each keyword.
        - `reddit_keywords_counts_month.csv`: counts # of keyword mentioning times over month.
- Step 3: Geolocation inference:
    - `python aggregate_smgeo.py`:
        - `reddit_geo_country_count.csv`: counts # of Reddit users from different countries.
        - `reddit_geo_state_count.csv`: counts # of Reddit users from different U.S. states.
    - `python aggregate_us_month.py`:
        - `reddit_{reddit_type}_keywords_geo_month.csv`: counts # of Reddit posts whose authors have geolocation inferred over month.
        - `reddit_{reddit_type}_keywords_us_month.csv`: counts # of Reddit posts whose authors are in U.S. over month.
  
### Subreddits list
A full list of relevant subreddits is [here](python/falconet/resources/annotators/subreddits/tobacco/e-cigs.txt).

 
### keywords list
Same as Twitter.

### Retrieving Reddit raw data
[Here](scripts/retrieve_reddit_data.py) is a script to retrieve Reddit data using [Retriever](https://github.com/kharrigian/retriever). 

## Contact
