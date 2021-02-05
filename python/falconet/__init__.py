from falconet import settings
from datetime import datetime


class Message(object):
    """
        Generic social media post
    """

    def __init__(self, p, use_previous_annotations=True):
        self.raw_data = p
        self.metadata = {}
        if use_previous_annotations and \
                isinstance(self.raw_data.get(settings.ANNOTATIONS_KEY, None), dict):
            self.annotations = self.raw_data[settings.ANNOTATIONS_KEY]
        else:
            self.annotations = {}

    def username(self):
        raise NotImplementedError

    def text(self):
        raise NotImplementedError

    def timestamp(self):
        raise NotImplementedError

    def __str__(self):
        out = "text: {}\nusername: {}\ntimestamp: {}\n\nannotations: {}\nmetadata: {}\n"
        return out.format(self.text().replace("\n", ""), self.username(), self.timestamp(),
                          repr(self.annotations),
                          repr(self.metadata))


class Tweet(Message):
    def __init__(self, p, use_previous_annotations=True):
        if not p.get(settings.TWEET_ID, None):
            raise ValueError("corrupted tweet")
        super(Tweet, self).__init__(p, use_previous_annotations=use_previous_annotations)

    def username(self):
        return self.raw_data["user"]["screen_name"]

    def text(self):
        if "extended_tweet" in self.raw_data and "full_text" in self.raw_data.get("extended_tweet"):
            return self.raw_data["extended_tweet"]["full_text"]
        return self.raw_data["text"]

    def timestamp(self):
        return self.raw_data[settings.TWEET_CREATED_AT]


class RedditComment(Message):
    def __init__(self, p, use_previous_annotations=True):
        super(RedditComment, self).__init__(p, use_previous_annotations=use_previous_annotations)

    def username(self):
        return self.raw_data["author"]

    def text(self):
        return self.raw_data["body"]

    def timestamp(self):
        return str(datetime.fromtimestamp(int(self.raw_data["created_utc"])))


class RedditSubmission(Message):
    def __init__(self, p, use_previous_annotations=True):
        super(RedditSubmission, self).__init__(p, use_previous_annotations=use_previous_annotations)

    def username(self):
        return self.raw_data["author"]

    def text(self):
        all_text = self.raw_data["title"]
        all_text += self.raw_data["selftext"] if self.raw_data["selftext"] else ''
        return all_text

    def timestamp(self):
        return str(datetime.fromtimestamp(int(self.raw_data["created_utc"])))
