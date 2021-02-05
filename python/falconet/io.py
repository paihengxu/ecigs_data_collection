from abc import ABC, abstractmethod
import bz2
import gzip
import lzma
import zlib
import logging
import os
from sys import stdout
try:
    import ujson as json
except:
    import json

from falconet import Tweet, RedditComment, RedditSubmission
from falconet.settings import ANNOTATIONS_KEY, TWEET_MINIMAL_FIELDS, ID, \
    TWEET_RETWEETED_STATUS
from falconet.utils import ensure_folder


class StreamReader(ABC):

    @abstractmethod
    def stream(self):
        raise NotImplementedError


# stub for a Mongo DB stream reader
class MongoDBStreamReader(StreamReader):

    def stream(self):
        raise NotImplementedError


# stub for a web API stream reader
class WebAPIStreamReader(StreamReader):

    def stream(self):
        raise NotImplementedError


class TextStreamReader(StreamReader):
    """
        Reads text from one or multiple files (in a folder) and returns Message objects
        Files are read lazily
    """
    def __init__(self, data_path, message_type, max_posts=None, use_previous_annotations=True):
        self.data_path = data_path
        self.message_type = message_type.lower()
        self.max_posts = max_posts
        self.use_previous_annotations = use_previous_annotations

        # define a function to generate the appropriate post type
        if self.message_type == "twitter":
            self.newMessage = self.__getTweet
        elif self.message_type == "reddit_comment":
            self.newMessage = self.__getRedditCcomment
        elif self.message_type == "reddit_submission":
            self.newMessage = self.__getRedditSubmission
        else:
            raise NotImplementedError("data source: {}".format(self.message_type))

    def __getTweet(self, m):
        return Tweet(m, use_previous_annotations=self.use_previous_annotations)

    def __getRedditCcomment(self, m):
        return RedditComment(m, use_previous_annotations=self.use_previous_annotations)

    def __getRedditSubmission(self, m):
        return RedditSubmission(m, use_previous_annotations=self.use_previous_annotations)

    def stream(self):
        fnames = []

        # loop through the input nargs
        for data_path_item in self.data_path:
            if os.path.isdir(data_path_item):
                print("[processing folder @ {}]".format(data_path_item))
                # process subfolders
                dir_list = [data_path_item]

                while len(dir_list) > 0:
                    data_dir = dir_list.pop()
                    flist = [
                        os.path.join(data_dir, f) for f in os.listdir(data_dir)]
                    for filep in flist:
                        if os.path.isdir(filep):
                            dir_list.append(filep)
                        else:
                            fnames.append(filep)
            else:
                # just one file
                fnames.append(data_path_item)

        for fname in fnames:
            print("[streaming from file @ {}]".format(fname))
            with open_file(fname, 'rt') as reader:
                try:
                    for num_posts, line in enumerate(reader):
                        # enough posts?
                        if self.max_posts and num_posts >= self.max_posts:
                            break
                        line = line.strip()
                        if len(line) == 0:
                            continue
                        try:
                            raw = json.loads(line)

                            if self.message_type == "twitter":
                                if "limit" in raw:
                                    # This is a Twitter limit message. Skip it.
                                    continue

                            # generate the appropriate message type
                            p = self.newMessage(raw)
                            yield p
                        except ValueError as e:
                            #print("invalid post?")
                            print (e)
                            continue
                except (OSError, EOFError, zlib.error) as e:
                    logging.warning('IOError: {}'.format(str(e)))


class StreamWriter:
    def __init__(self, filename, style=ID):
        self.filename = filename
        if filename and filename is not stdout:
            ensure_folder(filename)
            if filename[-3:] in ['.gz', 'bz2']:
                self.writer = open_file(filename, 'wb')
            else:
                self.writer = open_file(filename, 'w')
        else:
            self.writer = stdout

        if style == 'full':
            self.output_style = writer_style_full
        elif style == 'id':
            self.output_style = writer_style_id
        elif style == 'twitter_minimal':
            self.output_style = writer_style_twitter_minimal
        else:
            raise ValueError("Invalid writer style: " + str(style))

    def write(self, message):
        m = message.raw_data
        m = self.output_style(m)
        m[ANNOTATIONS_KEY] = message.annotations
        m = json.dumps(m) + '\n'
        if self.filename[-3:] in ['.gz', 'bz2']:
            m = m.encode('utf-8')
        self.writer.write(m)

    def close(self):
        if self.writer is not stdout:
            self.writer.close()


class MinimalWriter:
    def __init__(self, wpath, config):
        '''
        This class extracts defined a limited list of fields from annotated json entries.
        
        Parameters
        ----------
        wpath: str
            path to save extracted data
        config: [str | dict]
            A list of defined fields that allows for extracting json fields.
            Example: ["tweet_id:id", "user_id:user.id", "date:created_at", "keywords:annotations.keywords", "location_data:annotations.location"]
        '''
        self.wpath = wpath
        if self.wpath[-3:] in ['.gz', 'bz2']:
            self.wfile = open_file(self.wpath, 'wb')
        else:
            self.wfile = open_file(self.wpath, 'w')
        self.config = dict()

        # if no configuration, raise error
        if len(config) == 0:
            raise ValueError('Configuration can not be empty!')

        # check and parse the data
        for field in config:
            field = field.strip()
            
            # check field format
            if len(field) == 0:
                continue
            new_fields = field.split(':')
            if len(new_fields) != 2:
                raise ValueError('Format Error in: ', field)
            field_name = new_fields[0]
            new_fields = new_fields[1].split('.')
            if len(new_fields) == 0:
                raise ValueError(
                    'Format Error: configuration value type should be non-empty')

            if new_fields[0] != 'annotations' and new_fields[0] != 'metadata':
                new_fields = ['raw_data'] + new_fields
            self.config[field_name] = new_fields

    def write(self, message):
        entry = dict()
        for key in self.config:
            value = None

            # check if annotations, metadata or raw document
            if self.config[key][0] == 'annotations':
                value = message.annotations
            elif self.config[key][0] == 'metadata':
                value = message.metadata
            elif self.config[key][0] == 'raw_data':
                value = message.raw_data
            else:
                raise ValueError('The first path key must be one of the three options: annotations, metadata, raw_data!')

            # if the path does not exist, assign None as default
            for path_key in self.config[key][1:]:
                if path_key in value:
                    value = value[path_key]
                    
                    # special design for the location, only keep country, state and city
                    if path_key == 'location' and type(value) == dict:
                        value = {
                            'country': value.get('country', None),
                            'state': value.get('state', None),
                            'city': value.get('city', None)
                        }
                        for tmp_key in value:
                            if len(value[tmp_key]) == 0:
                                value[tmp_key] = None
                else:
                    value = None
                    break
            # record value
            entry[key] = value

        # write to file
        entry = json.dumps(entry)+'\n'
        if self.wpath[-3:] in ['.gz', 'bz2']:
            entry = entry.encode('utf-8')
        self.wfile.write(entry)

    def close(self):
        self.wfile.flush()
        self.wfile.close()


def open_file(filename, mode, **kwargs):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode=mode, **kwargs)
    elif filename.endswith('.bz2'):
        return bz2.open(filename, mode=mode, **kwargs)
    elif filename.endswith('.xz'):
        return lzma.open(filename, mode=mode, **kwargs)
    elif filename.endswith('.zst'):
        raise NotImplementedError
    return open(filename, mode=mode, **kwargs)


def writer_style_full(m):
    return m


def writer_style_id(m):
    new_m = dict()
    new_m[ID] = m[ID]
    return new_m


def writer_style_twitter_minimal(tweet):
    new_tweet = {}
    check_and_add_field(tweet, new_tweet, TWEET_MINIMAL_FIELDS)
    return new_tweet


def check_and_add_field(tweet, new_tweet, keys_to_keep):
    for key, value in tweet.items():
        if key in keys_to_keep:
            if key == TWEET_RETWEETED_STATUS:
                if isinstance(value, dict):
                    new_value = {}
                    check_and_add_field(value, new_value, TWEET_MINIMAL_FIELDS)
                    new_tweet[key] = new_value
            elif isinstance(value, dict):
                if keys_to_keep[key] is True:
                    new_tweet[key] = value
                else:
                    new_value = {}
                    check_and_add_field(value, new_value, keys_to_keep[key])
                    new_tweet[key] = new_value
            else:
                new_tweet[key] = value
