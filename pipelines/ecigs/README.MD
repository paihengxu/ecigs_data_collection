# E-cigarettes Data Collection Pipeline
This README contains descriptions for scripts that are relevant to the created ecigs pipeline.  
The instruction for running the pipeline files are provided in the [main README](../../README.MD).

## Twitter

### Pipeline files
`twitter_keywords_pipeline.json` and `twitter_relevance_pipeline.json`: pipeline configuration json files for Twitter.

`run_twitter_pipeline.py`: the script to run Twitter pipeline.

### Twitter Aggregation scripts
Here are the scripts to help extract information from the output csv files. 
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

 
## Reddit

### Pipeline files:
`reddit_subreddit_pipeline.json` and `reddit_keywords_pipeline.json`: pipeline configuration json files for Reddit.

`run_reddit_pipeline.py`: the script to run Reddit pipeline.

`run_reddit_geo.py` and `smgeo_infer.sh`: the script to run Reddit geolocation inference.

### Reddit Aggregation scripts
To associate users' geolocation information with Reddit posts (submissions or comments), 
run the following command to count the number of posts with geolocation information or in United States.

`python count_reddit_geo.py --fn {input_fn} --outfn {output}`

where

- `{input_fn}`: Keyword filtered data file path, `.gz` files in `{REDDIT_PROCESSED_DATA_DIR}/reddit/reddit_{reddit_type}/output_reddit_keywords/`.
- `{output}`: output file name, in `.csv` format. 

The output files will be in the same directory as `{input_fn}`, and create the following two subdirectories:

- Count of posts with geolocation information will be in `geo_count_month`,
- Counts of posts in U.S. will be in `us_count_month`.

Similar to Twitter, here are the scripts to help extract information from the output csv files:

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