import sys
import os
from pathlib import Path

from falconet.settings.local import *

TIMESTAMP = "timestamp"
ANNOTATIONS_KEY = 'annotations'
TOKENS_KEY = 'tokens'
LOCATION_KEY = 'location'
METADATA_ANNOTATIONS_KEY = 'metadata_annotations'
TOKENS_LOWER_KEY = 'tokens_lower'
TWEET_ID = 'id'
TWEET_USER = 'user'
TWEET_USER_VERIFIED = 'verified'
TWEET_USER_VERIFIED_ANNOTATION_KEY = 'verified-user'
TWEET_ENTITIES = 'entities'
TWEET_ENTITIES_URLS = 'urls'
SHARES_LINK_ANNOTATION_KEY = 'shares-link'
ID = 'id'
TWEET_CREATED_AT = 'created_at'
TWEET_TEXT = 'text'
TWEET_RETWEETED_STATUS = 'retweeted'
COUNTER_TOTAL_KEY = 'total'
DATE_KEY = 'date'
COUNTER_TOTAL_GEOLOCATION_KEY = 'total_geolocated'
COUNTER_AGGREGATE_LOCATION_ATTRIBUTES = ('country', 'state', 'id')
PIPELINE_NAME_KEY = 'pipeline'
OUTPUT_FORMAT_JSON = 'json'
OUTPUT_FORMAT_POSTGRESQL = 'postgresql'
LOCATIONS_KEY = 'locations'
GAP_IN_SECONDS = 3600

PUBLIC_ANNOTATION_INDEX = -1
HEALTH_ANNOTATION_INDEX = -2
HEALTH_STREAM_ANNOTATION_INDEX = -3
HEALTH_GEOLOCATED_ANNOTATION_INDEX = -4
PUBLIC_GEOLOCATED_ANNOTATION_INDEX = -5
LOCATION_ANYWHERE_ID = -1

# TODO: which one is used in Twitter date field?
# TWITTER_DATE_FORMAT = '%m/%d/%Y %H:%M:%S +0000'
TWITTER_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
COUNTER_DATE_FORMAT = '%Y/%m/%d'
COUNTER_HOUR_FORMAT = '%Y/%m/%d %H:00:00'

directory = Path(sys.modules[__package__].__file__).parent.parent
RESOURCES_DISEASE_KEYWORDS = os.path.join(directory, 'resources/annotators/keywords/')
RESOURCES_SUBREDDITS = os.path.join(directory, 'resources/annotators/subreddits/tobacco/')

TWEET_MINIMAL_FIELDS = {
                    'created_at': True,
                    'text': True,
                    'id': True,
                    'retweeted': True,
                    'user': {
                        'location': True,
                        'utc_offset': True,
                        'time_zone': True,
                        'name': True,
                        'description': True,
                        'created_at': True,
                        'id': True,
                        'screen_name': True,
                        'lang': True,
                        'geo_enabled': True,
                    },
                    'place': {
                        'full_name': True,
                        'url': True,
                        'country': True,
                        'place_type': True,
                        'id': True,
                        'name': True,
                        'country_code': True,
                        'attributes': True,
                        'bounding_box': True,
                    },
                    'geo': {
                        'coordinates': True,
                    },
                    'coordinates': {
                        'coordinates': True,
                    },
                }

# Config variables for classifiers ###
EMBEDDINGS_URL = 'http://nlp.stanford.edu/data/glove.twitter.27B.zip'