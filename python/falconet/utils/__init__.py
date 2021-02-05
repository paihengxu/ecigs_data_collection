#!/usr/bin/env python
# -*- coding: utf-8 -*-

from requests_oauthlib import OAuth2Session
# import falconet.settings
import datetime, time, pytz
from dateutil import parser,tz
import os


class BitbucketClientSecrets:
    """
    Shamelessly stolen from https://developer.atlassian.com/blog/2016/02/bitbucket-oauth-with-python/

    The structure of this class follows Google convention for `client_secrets.json`:
    https://developers.google.com/api-client-library/python/guide/aaa_client_secrets
    Bitbucket does not emit this structure so it must be manually constructed.
    """
    
    def __init__(self, key, secret):
        self.client_id = key
        self.client_secret = secret
        self.redirect_uris = ["https://localhost"]
        self.auth_uri = "https://bitbucket.org/site/oauth2/authorize"
        self.token_uri = "https://bitbucket.org/site/oauth2/access_token"
        self.base_uri = "https://api.bitbucket.org/2.0/repositories/mdredze/falconet/downloads/"


def connect_to_bitbucket(key, secret):
    """
    To automatically download data files from bitbucket repo.  User needs a bitbucket
    account with access to the falconet project
    """
    c = BitbucketClientSecrets(key, secret)
    # Fetch a request token
    bitbucket = OAuth2Session(c.client_id)
    print('Key:', c.client_id)
    print('Secret:', c.client_secret)
    # Redirect user to Bitbucket for authorization
    authorization_url = bitbucket.authorization_url(c.auth_uri)
    print('Please go here and authorize: {}'.format(authorization_url[0]))
    # Get the authorization verifier code from the callback url
    redirect_response = input('Paste the full redirect URL here: ')
    # Fetch the access token
    bitbucket.fetch_token(
        c.token_uri,
        authorization_response=redirect_response,
        username=c.client_id,
        password=c.client_secret)
    return bitbucket, c.base_uri

class Timer:
    def __init__(self):
        self.elapsed = 0
        self.start_time = time.time()

    def start(self):
        self.start_time = time.time()
        self.elapsed = 0

    def stop(self):
        current = time.time()
        self.elapsed += current - self.start_time

    def resume(self):
        self.start_time = time.time()

    def get_full_time(self):
        return format_full_time(self.elapsed * 1000)

def get_hour_str_from_date_with_utc_offset(date_str,time_zone="UTC"):
    '''Returns the hour string %Y-%m-%d %H:00:00 %z for given input datestring with utc_off_set, default time_zone = UTC'''
    tzlocal = tz.gettz(time_zone)
    z = (parser.parse(date_str))
    # print('orignal z is ',z)
    # print('z converted is is',z.astimezone(tzlocal))
    # print('z is without offset',z.tzinfo.utcoffset(d))

    coverted_time_zone = z.astimezone(tzlocal)
    hour_str = coverted_time_zone.strftime('%Y-%m-%d %H:00:00 %z' )

    return hour_str

def get_day_from_date(date_str, time_zone="UTC"):
    '''Returns the day string %Y-%m-%d for given input datestring with utc_off_set, default time_zone = UTC'''

    tzlocal = tz.gettz(time_zone)
    z = (parser.parse(date_str))
    coverted_time_zone = z.astimezone(tzlocal)
    day_str = coverted_time_zone.strftime('%Y-%m-%d' )

    return day_str

def get_month_from_date(date_str, time_zone='UTC'):
    '''
    Returns the month string %Y-%m for given input datestring with utc_off_set, default time_zone = UTC
    '''
    tzlocal = tz.gettz(time_zone)
    z = (parser.parse(date_str))
    coverted_time_zone = z.astimezone(tzlocal)
    month_str = coverted_time_zone.strftime('%Y-%m')

    return month_str

def get_timestamp_from_date(date_str):
    timestamp = time.mktime(datetime.datetime.strptime(date_str, '%a %b %d %H:%M:%S +0000 %Y').timetuple())
    return timestamp


def format_full_time(elapsed):
    ms = int(elapsed)
    seconds = int(elapsed / 1000)
    ms %= 1000
    minutes = int(seconds / 60)
    seconds %= 60
    hours = int(minutes / 60)
    minutes %= 60
    days = int(hours / 24)
    hours %= 24
    
    msg = []
    if days != 0:
        msg.append(days)
        msg.append('days')
    if hours != 0:
        msg.append(hours)
        msg.append('hours')
    if minutes != 0:
        msg.append(minutes)
        msg.append('minutes')
    if seconds != 0:
        msg.append(seconds)
        msg.append('seconds')
    if ms != 0:
        msg.append(ms)
        msg.append('ms')
    
    return ' '.join(map(str, msg))

def ensure_folder(path):
    # create folder if does not exist
    folder = os.path.dirname(path)
    if not os.path.exists(folder):
        os.makedirs(folder)     


class Alphabet:
    def __init__(self):
        self._wToI = {}
        self._iToW = {}
        self.__size = 0
        self.isFixed = False

    def wtoi(self, w):
        if w in self._wToI:
            return self._wToI[w]
        elif not self.isFixed:
            self._wToI[w] = self.__size
            self._iToW[self.__size] = w
            self.__size += 1
            return self._wToI[w]
        else:
            return None

    def put(self, w):
        return self.wtoi(w)

    def itow(self, i):
        if i not in self._iToW:
            return None
        else:
            return self._iToW[i]

    def get(self, i):
        return self.itow(i)

    def __len__(self):
        return self.__size


def fatal_exception(message, exception, logger):
    logger.fatal(message)
    raise exception(message)


def str_to_bool(my_str):
    """helper to return True or False based on common values"""
    my_str = str(my_str).lower()
    if my_str in ['yes', 'y', 'true', '1']:
        return True
    elif my_str in ['no', 'n', 'false', '0']:
        return False
    else:
        raise ValueError("unknown string for bool conversion:%s".format(my_str))
