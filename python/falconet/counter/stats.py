import copy
from datetime import datetime
import pytz
from falconet.settings import *


class Stats:
    def __init__(self, json_object=None, location_id=None, day=None, timestamp=None, timezone=None):
        if json_object:
            # Twitter uses UTC timezone
            self.datetime = pytz.utc.localize(datetime.strptime(json_object[DATE_KEY], TWITTER_DATE_FORMAT))
            if timezone:
                self.datetime = self.datetime.astimezone(timezone)
            
            self.day = self.datetime.date()
            self.pipeline = self.__get_value(json_object, PIPELINE_NAME_KEY)
            self.location_id = self.__get_value(json_object, LOCATION_KEY) 
            self.total = self.__get_value(json_object, COUNTER_TOTAL_KEY) 
            self.annotations = self.__get_value(json_object, ANNOTATIONS_KEY) 
            self.total_geolocated = self.__get_value(json_object, COUNTER_TOTAL_GEOLOCATION_KEY) 
        else:
            self.location_id = location_id
            self.day = day
            self.datetime = timestamp
            self.total = None
            self.total_geolocated = None
            self.annotations = None
            self.pipeline = None
    
    def to_str(self):
        return '\t'.join(map(str, [self.datetime, self.pipeline, self.location_id, 
                                   self.total, self.total_geolocated, self.annotations]))

    def __get_value(self, json_obj, key):
        if key in json_obj:
            return json_obj[key]
        else:
            return None 
    
    def copy_stats_to_parent(self, parent_location_id):
        parent_stat = copy.deepcopy(self)
        parent_stat.location_id = parent_location_id
        return parent_stat
    

class CombinedStats:
    def __init__(self, timestamp=None, pipeline=None):
        self.timestamp = timestamp
        self.pipeline = pipeline
        self.total = 0
        self.total_geolocated = 0
        self.location_counter = {}
        self.annotation_location_counter = {}
