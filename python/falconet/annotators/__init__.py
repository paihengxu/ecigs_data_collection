import re
import sys
import langid
from urllib.error import HTTPError
try:
    from ipdb import set_trace
except ImportError:
    from pdb import set_trace
from flashtext import KeywordProcessor
import carmen

from falconet.tokenizer import tokenize_raw_text
import falconet
from falconet import settings
from falconet.classifiers.classifier import TextTokenClassifier


class Annotator:
    """ Object that processes a tweet and adds annotations or metadata to it. """

    def annotate(self, message):
        """
        Parameters
        ----------
        tweet : { str:Object }
            tweet, JSON object
        annotations : { str:Object }
            labels applied to this tweet
        metadata : { str:Object }
            additional info about tweet, avoids redundant processing         
        """
        raise NotImplementedError


class ClassifierAnnotator(Annotator):
    """ Adds the predicted label to each tweet.  Assumes text is tokenized and sitting in metadata. """

    def __init__(self, model_path, model_name_prefix=None, skip_labels=None):
        '''

        Parameters
        ----------
        model_path: str
            path of TextTokenClassifier model
        model_name_prefix: str
            optional, model_name_prefix
        skip_labels: list
            optional, a list of labels should skip
        '''
        self._model = TextTokenClassifier.deserialize(model_path)
        self.skip_labels = skip_labels

        if model_name_prefix is None:
            self._modelNamePrefix = 'label-'
        else:
            self._modelNamePrefix = model_name_prefix

    def annotate(self, message):
        if settings.TOKENS_LOWER_KEY not in message.metadata:
            raise ValueError('It appears TokenizeAnnotator has not been run in this pipeline.')

        label = self._model.predict(
            [message.metadata[settings.TOKENS_LOWER_KEY]],
            pretokenized=True
        )[0]

        # check if skip
        if self.skip_labels and label in self.skip_labels:
            if 'skip_message' not in message.metadata:
                message.metadata['skip_message'] = dict()
            message.metadata['skip_message']['label'] = True
        else:
            message.annotations[self._modelNamePrefix + self._model._name] = label


class ScoreAnnotator(Annotator):
    """ Adds a dictionary of label scores to each tweet """

    def __init__(self, **kwargs):
        '''

        Parameters
        ----------
        model_path: str
            Model path
        model_name_prefix: str
            Model name prefix
        skip_threshold: float
            Exclusive probability threhold to filter out the scores,
            keep only ones larger than the threhold.
        '''
        self._model = TextTokenClassifier.deserialize(kwargs.get('model_path'))
        self.skip_threshold = kwargs.get('skip_threshold', float('-inf'))

        model_name_prefix = kwargs.get('model_name_prefix', None)
        if model_name_prefix is None:
            self._modelNamePrefix = 'score-'
        else:
            self._modelNamePrefix = model_name_prefix

    def annotate(self, message):
        if settings.TOKENS_LOWER_KEY not in message.metadata:
            raise ValueError('It appears TokenizeAnnotator has not been run in this pipeline.')

        # score is a dictionary of the most likely label and its probability
        score = self._model.confidence(
            [message.metadata[settings.TOKENS_LOWER_KEY]], pretokenized=True
        )[0]

        # check if skip
        if self.skip_threshold and list(score.values())[0] <= self.skip_threshold:
            if 'skip_message' not in message.metadata:
                message.metadata['skip_message'] = dict()
            message.metadata['skip_message']['confidence'] = 'LOWER_THAN_THRESHOLD'
        else:
            message.annotations[self._modelNamePrefix + self._model._name] = score


class KeywordAnnotator(Annotator):
    def __init__(self, **kwargs):
        self.label = kwargs.get('label', True)
        filename_to_load = kwargs.get('filename', None)
        self.skip_misses = kwargs.get('skip_misses', True)
        self.include_hashtags = kwargs.get('include_hashtags', True)

        # Load the keywords
        '''keywords = []
        with open(filename_to_load) as my_input:
            for line in my_input:
                phrase = line.strip().lower()
                keywords.append(phrase)

                if self.include_hashtags and not phrase.startswith('#'):
                    phrase = '#' + phrase
                    keywords.append(phrase)

        self.trie = Trie(keywords)'''

        # flashtext implementation
        self.trie = KeywordProcessor()
        with open(filename_to_load) as my_input:
            for line in my_input:
                phrase = line.strip().lower()
                self.trie.add_keyword(phrase)
                if self.include_hashtags and not phrase.startswith('#'):
                    phrase = '#' + phrase
                    self.trie.add_keyword(phrase)

    def annotate(self, message):
        if not settings.TOKENS_LOWER_KEY in message.metadata:
            raise ValueError('It appears TokenizeAnnotator has not been run in this pipeline.')
        tokens = message.metadata[settings.TOKENS_LOWER_KEY]
        match = False
        keywords = []
        # Check all substrings
        #for pos, word in enumerate(tokens):
            #if self.trie.has_keys_with_prefix(word):
                # There is a phrase that starts with this word.
                # Is this the entire phrase?
                #if word in self.trie:
                    #keywords.append(word)
                    #match = True
                # We need to try a longer sequence. For simplicity, try the full text
                # constructed from this point to the end of the message
                # In reality, we rarely see longer phrases so its pretty efficient to construct
                # this case separately.
                # text_to_end = ' '.join(tokens[pos:])
                # matches = self.trie.prefixes(text_to_end)
                # if len(matches) > 0:
                #    keywords.append(text_to_end)
                #    match = True

        # flashtext implementation
        keywords = self.trie.extract_keywords(' '.join(tokens))
        if len(keywords) > 0:
            match = True

        if match:
            message.annotations[self.label] = True
            message.annotations['keywords'] = keywords
        else:
            message.annotations[self.label] = False
            if self.skip_misses:
                if 'skip_message' not in message.metadata:
                    message.metadata['skip_message'] = dict()
                message.metadata['skip_message']['keyword'] = 'no_keyword_found'.upper()


class TokenizeAnnotator(Annotator):
    def annotate(self, message):
        tokens = tokenize_raw_text(message.text())
        message.metadata[settings.TOKENS_KEY] = tokens
        message.metadata[settings.TOKENS_LOWER_KEY] = [
            token.lower() for token in tokens]


class LangidAnnotator(Annotator):
    def __init__(self, **kwargs):
        '''

        Parameters
        ----------
        skip_unless_language: list
            The list of languages to run pipe_line on,
            keeps all if default
            Provide iso codes, see: 639-1 codes
            link to the repo: https://github.com/saffsd/langid.py
        '''
        self.skip_unless_language = kwargs.get('skip_unless_language', None)

    def annotate(self, message):
        # preprocess text for a better classification, user mentions, URLs, hashtags
        new_text = message.text()
        new_text = re.sub(r'#\S+', '', re.sub(
            r'@\w+', '', re.sub(
                r'https?:\S+', '', new_text)
            )
        )
        # detect language
        tweet_langid, _ = langid.classify(new_text)

        if self.skip_unless_language and tweet_langid not in self.skip_unless_language:
            if 'skip_message' not in message.metadata:
                message.metadata['skip_message'] = dict()
            message.metadata['skip_message']['lang'] = "Skipped lang: {}".format(
                tweet_langid).upper()
        else:
            message.annotations['langid'] = {'lang': tweet_langid}


class CarmenLocationAnnotator(Annotator):
    def __init__(self, **kwargs):
        '''
        Initialization for CarmenLocationAnnotator

        Parameters
        ----------
        keep_location: dict
            A dictionary of location attributes (country, state, county, city).
            Within each attribute, the value is a list of defined values.
            Example {'country': ['United States'], 'state': ['Maryland', 'Colorado']}
        '''
        self.carmen = carmen.get_resolver()
        self.carmen.load_locations()
        self.keep_location = kwargs.get('keep_location', None)

    def get_annotation_type(self):
        return settings.LOCATION_KEY

    def annotate(self, tweet):
        assert isinstance(tweet, falconet.Tweet), "Carmen only works with Tweets"

        location = self.carmen.resolve_tweet(tweet.raw_data)
        # default to None
        tweet.annotations["location"] = None
        if location:
            location = location[1]

            # check if the location is in the
            skip_flag = False
            skip_messages = dict() # each key is location level, value is the skip message

            if self.keep_location:
                # loop through the location keys: country, state, county, city
                for loc_key in self.keep_location:
                    if type(self.keep_location[loc_key]) == list and \
                            len(self.keep_location[loc_key]) > 0:
                        # to ensure the user defined location levels are in the Location object
                        if loc_key in location.__dict__:
                            # check if the location[loc_key] not None
                            # check the location is in the keep list
                            if location.__dict__[loc_key] and \
                                    location.__dict__[loc_key] not in self.keep_location[loc_key]:
                                # skip message format: location level + skipped location name
                                skip_flag = True
                                skip_messages[loc_key] = 'INVALID_{0}: {1}'.format(
                                    loc_key.upper(), str(location.__dict__[loc_key])
                                )

            # if skip flag is true
            if skip_flag:
                if 'skip_message' not in tweet.metadata:
                    tweet.metadata['skip_message'] = dict()
                tweet.metadata['skip_message']['location'] = skip_messages
            elif location:
                tweet.annotations["location"] = location.__dict__


class SubredditAnnotator(Annotator):
    """
    takes in a list of subreddits
    """

    def __init__(self, **kwargs):
        kw_path = kwargs['kw_path']
        subreddits = set()
        with open(kw_path, 'r') as inf:
            for line in inf:
                subreddits.add(line.strip())
        self.subreddits = subreddits

    def annotate(self, message):
        assert isinstance(message, falconet.RedditSubmission) or isinstance(message, falconet.RedditComment), \
            "SubredditAnnotator only supports Reddit submissions and comments."
        subreddit = message.raw_data.get('subreddit', 'UNK')
        if subreddit in self.subreddits:
            message.annotations['subreddit'] = subreddit
        else:
            message.metadata['skip_message'] = 'invalid_subreddit'
