import ujson as json

import carmen

from falconet.utils import Timer, format_full_time
from falconet.io import open_file
from falconet.counter.stats import Stats, CombinedStats
from falconet import settings


class DataExporter:
    HOURLY_STATS = 'hourly_stats'
    DAILY_STATS = 'daily_stats'
    DAY = 'date'
    HOUR = 'hour'

    def __init__(self, timezone):
        self.__annotation_name_to_index = dict()
        self.__location_resolver = carmen.get_resolver()
        self.__location_resolver.load_locations()
        self.__load_annotation_names(settings.RESOURCES_ANNOTATION_NAMES)
        self.__timezone = timezone
        self.__output_format = None
        self.__output_writer = None
        self.__gap_writer = None
    
    def __load_annotation_names(self, file_name):
        with open_file(file_name, 'r') as f:
            for line in f:
                split_strings = line.strip().split('\t')
                annotation_key = split_strings[1]
                if annotation_key.endswith('_key'):
                    annotation_key = annotation_key[0:-4]
                annotation_index = int(split_strings[0])
                self.__annotation_name_to_index[annotation_key] = annotation_index

    def run(self, input_path, output_file, output_format, gap_file, load_after_date, 
            output_after_date):
        timer = Timer()
        timer.start()
        
        self.__output_format = output_format
        self.__output_writer = open_file(output_file, 'w')
        self.__gap_writer = open_file(gap_file, 'w')
        print('Collecting files: ' + input_path)
        files = self.__collect_files(input_path, load_after_date)
        print('Found ' + str(len(files)) + ' count files.')
        
        self.__process_files(files, output_after_date)
        self.__output_writer.close()
        self.__gap_writer.close()
        timer.stop()
        
        print('Finished.')
        print('Used ' + timer.get_full_time() + '\n')

    def __process_files(self, files, output_after_date):
        self.__daily_health_counter = dict()
        self.__daily_public_counter = dict()
        self.__hourly_health_counter = dict()
        self.__hourly_public_counter = dict()
        
        self.__day_set = set()
        self.__hour_set = set()
        
        for file_name in files:
            print(file_name)
            with open_file(file_name, 'rb') as f:
                for line in f:
                    line = line.strip()
                    if line != '':
                        stats = Stats(json_object=json.loads(line), timezone=self.__timezone)
                        self.__hour_set.add(stats.datetime)
                        self.__day_set.add(stats.day)
                        if output_after_date is None or stats.datetime > output_after_date:
                            if stats.location_id is not None and stats.location_id != -1:
                                parents_stats = list()
                                parents_stats.append(stats)
                                self.__resolve_parent_locations(stats, parents_stats)
                            
                                for stats in parents_stats:
                                    self.__combine_stats(stats, DataExporter.HOURLY_STATS)
                                    self.__combine_stats(stats, DataExporter.DAILY_STATS)
                            else:
                                self.__combine_stats(stats, DataExporter.HOURLY_STATS)
                                self.__combine_stats(stats, DataExporter.DAILY_STATS)
        
        sorted_hours = sorted(self.__hour_set)
        sorted_days = sorted(self.__day_set)
        
        # Look for gaps
        self.__save_gap(sorted_hours)
            
        if self.__output_format == settings.OUTPUT_FORMAT_JSON:
            self.__save_stats_as_json(sorted_days, DataExporter.DAY,
                                      self.__daily_public_counter, self.__daily_health_counter)
            self.__save_stats_as_json(sorted_hours, DataExporter.HOUR,
                                      self.__hourly_public_counter, self.__hourly_health_counter)
        else:
            self.__save_daily_stats_as_postgresql(sorted_days)

    def __resolve_parent_locations(self, stats, stats_list):
        if stats.location_id not in self.__location_resolver.location_id_to_location:
            print(str(stats.location_id) + ' is not in carmen')
            return
        location = self.__location_resolver.get_location_by_id(stats.location_id)
        
        if location.parent_id != -1 and location.parent_id != stats.location_id:
            parent_location = self.__location_resolver.get_location_by_id(location.parent_id)
            parent_stat = stats.copy_stats_to_parent(parent_location.id)
            stats_list.append(parent_stat)
            self.__resolve_parent_locations(parent_stat, stats_list)
            # print('parent ' + str(parent_location) + ' child ' + str(stats.location_id))
            pass

    def __save_gap(self, sorted_hours):
        previous = None
        for current in sorted_hours:
            if previous: 
                gap = (current - previous).total_seconds()
                if gap > settings.GAP_IN_SECONDS:
                    gap_string = format_full_time(gap*1000)
                    print(previous)
                    gap_info = '\t'.join(map(str, [previous, current, gap_string]))
                    print('Found gap: ' + gap_info)
                    self.__gap_writer.write(gap_info + '\n')
            previous = current

    def __format_timestamp(self, timestamp, timestamp_type):
        if timestamp_type == DataExporter.DAY:
            return timestamp.strftime(settings.COUNTER_DATE_FORMAT)
        else:
            return timestamp.strftime(settings.COUNTER_HOUR_FORMAT)

    def __save_stats_as_json(self, sorted_timestamps, timestamp_type, 
                             public_counter, health_counter):
        """ Write the daily and hourly stats as json file.
        """
        
        for timestamp in sorted_timestamps:
            timestamp_string = self.__format_timestamp(timestamp, timestamp_type)
            print('Saving stats for ' + timestamp_string)
            
            result = dict()
            result[timestamp_type] = timestamp_string
           
            if timestamp in public_counter:
                stats = public_counter[timestamp]
                
                public_result = dict()
                public_result[settings.COUNTER_TOTAL_GEOLOCATION_KEY] = stats.total_geolocated
                
                locations = []
                for location in stats.location_counter:
                    locations.append({settings.LOCATION_KEY: location,
                                      settings.COUNTER_TOTAL_KEY: stats.location_counter[location].total})
                public_result[settings.LOCATIONS_KEY] = locations
                result[settings.PUBLIC_PIPELINE] = public_result
            
            if timestamp in health_counter:
                stats = health_counter[timestamp]
                
                health_result = dict()
                # Total number of tweets in health stream
                health_result[settings.COUNTER_TOTAL_KEY] = stats.total
                # Total number of geo-located tweets in health stream
                health_result[settings.COUNTER_TOTAL_GEOLOCATION_KEY] = stats.total_geolocated
                
                # Total number of health tweets per location
                locations = []
                for location, count in stats.location_counter.items():
                    locations.append({settings.LOCATION_KEY: location, settings.COUNTER_TOTAL_KEY: count})
                health_result[settings.LOCATIONS_KEY] = locations
                
                # Number of tweets per location per category
                annotations = {}
                for annotation_name, location_counter in stats.annotation_location_counter.items():
                    if annotation_name == settings.ANNOTATION_HEALTH:
                        continue
                    if annotation_name not in self.__annotation_name_to_index:
                        print('Missing index for annotation name: ' + annotation_name)
                    else:
                        locations = []
                        for location, count in location_counter.items():
                            locations.append({settings.LOCATION_KEY: location, settings.COUNTER_TOTAL_KEY: count})
                        annotations[annotation_name] = locations
                health_result[settings.ANNOTATIONS_KEY] = annotations
                result[settings.HEALTH_PIPELINE] = health_result
            self.__output_writer.write(json.dumps(result) + '\n')

    def __save_daily_stats_as_postgresql(self, sorted_days):
        """ Write the daily stats as postgresql file. The format of each line is:
            annotation (int) location (int) date (date) value (int)
        """
        for day in sorted_days:
            date_string = day.strftime(settings.COUNTER_DATE_FORMAT)
            print('Saving stats for ' + date_string)
            
            if day in self.__daily_public_counter:
                daily_stats = self.__daily_public_counter[day]
                # Total number of geolocated tweets in public stream:
                msg = [settings.PUBLIC_GEOLOCATED_ANNOTATION_INDEX,
                       settings.LOCATION_ANYWHERE_ID, date_string,
                       daily_stats.total_geolocated]
                self.__output_writer.write('\t'.join(map(str, msg)) + '\n')
                
                # Total number of public tweets per location
                for location in daily_stats.location_counter:
                    msg = [settings.PUBLIC_ANNOTATION_INDEX, location, date_string,
                           daily_stats.location_counter[location].total]
                    self.__output_writer.write('\t'.join(map(str, msg)) + '\n')
            
            if day in self.__daily_health_counter:
                daily_stats = self.__daily_health_counter[day]
                
                # Total number of tweets in health stream
                msg = [settings.HEALTH_STREAM_ANNOTATION_INDEX, settings.LOCATION_ANYWHERE_ID,
                       date_string, daily_stats.total]
                self.__output_writer.write('\t'.join(map(str, msg)) + '\n')
                
                # Total number of geo-located tweets in health stream
                msg = [settings.HEALTH_GEOLOCATED_ANNOTATION_INDEX, settings.LOCATION_ANYWHERE_ID,
                       date_string, daily_stats.total_geolocated]
                self.__output_writer.write('\t'.join(map(str, msg)) + '\n')
                
                # Total number of health tweets per location
                for location, count in daily_stats.location_counter.items():
                    msg = [settings.HEALTH_ANNOTATION_INDEX, location, date_string, count]
                    self.__output_writer.write('\t'.join(map(str, msg)) + '\n')
                
                # Number of tweets per location per category
                for annotation_name, location_counter in daily_stats.annotation_location_counter.items():
                    # if annotation_name == ANNOTATION_HEALTH:
                    #    continue
                    if annotation_name not in self.__annotation_name_to_index:
                        print('Missing index for annotation name: ' + annotation_name)
                    else:
                        annotation_id = self.__annotation_name_to_index[annotation_name]
                        for location, count in location_counter.items():
                            msg = [annotation_id, location, date_string, count]
                            self.__output_writer.write('\t'.join(map(str, msg)) + '\n')

    def __combine_stats(self, stats, stats_type):
        counter = None
        if stats_type == DataExporter.DAILY_STATS:
            stats_timestamp = stats.day
            if stats.pipeline == settings.HEALTH_PIPELINE:
                counter = self.__daily_health_counter
            elif stats.pipeline == settings.PUBLIC_PIPELINE:
                counter = self.__daily_public_counter
        elif stats_type == DataExporter.HOURLY_STATS:
            stats_timestamp = stats.datetime
            if stats.pipeline == settings.HEALTH_PIPELINE:
                counter = self.__hourly_health_counter
            elif stats.pipeline == settings.PUBLIC_PIPELINE:
                counter = self.__hourly_public_counter
        else:
            raise ValueError("invalid stats_type:%s" % stats_type)

        if counter is None:
            return

        if stats_timestamp in counter:
            combined_stats = counter[stats_timestamp]
        else:
            combined_stats = CombinedStats(timestamp=stats_timestamp, pipeline=stats.pipeline)
        
        if stats.total is not None and stats.total_geolocated is not None:
            combined_stats.total += stats.total
            combined_stats.total_geolocated += stats.total_geolocated
            if stats.annotations:
                for annotation_name, annotation_count in stats.annotations.items():
                    if annotation_name == settings.ANNOTATION_HEALTH:
                        if settings.LOCATION_ANYWHERE_ID in combined_stats.location_counter:
                            combined_stats.location_counter[settings.LOCATION_ANYWHERE_ID] += annotation_count
                        else:
                            combined_stats.location_counter[settings.LOCATION_ANYWHERE_ID] = annotation_count
                    else:
                        if annotation_name in combined_stats.annotation_location_counter:
                            location_counter = combined_stats.annotation_location_counter[annotation_name]
                        else:
                            location_counter = combined_stats.annotation_location_counter[annotation_name] = {}
                        
                        if settings.LOCATION_ANYWHERE_ID in location_counter:
                            location_counter[settings.LOCATION_ANYWHERE_ID] += annotation_count
                        else:
                            location_counter[settings.LOCATION_ANYWHERE_ID] = annotation_count
                        combined_stats.annotation_location_counter[annotation_name] = location_counter
                    
        elif stats.annotations is not None and len(stats.annotations) > 0:
            for annotation_name, annotation_count in stats.annotations.items():
                if annotation_name == settings.ANNOTATION_HEALTH:
                    if stats.location_id in combined_stats.location_counter:
                        combined_stats.location_counter[stats.location_id] += annotation_count
                    else:
                        combined_stats.location_counter[stats.location_id] = annotation_count
                else:
                    if annotation_name in combined_stats.annotation_location_counter:
                        location_counter = combined_stats.annotation_location_counter[annotation_name]
                    else:
                        location_counter = combined_stats.annotation_location_counter[annotation_name] = {}
                    
                    if stats.location_id in location_counter:
                        location_counter[stats.location_id] += annotation_count
                    else:
                        location_counter[stats.location_id] = annotation_count
                    combined_stats.annotation_location_counter[annotation_name] = location_counter
        counter[stats_timestamp] = combined_stats

    def __collect_files(self, input_path, load_after_date=None):
        files = []
        DataExporter.walkin(input_path, files)
        
        # If the file names follow a date format: 
        if load_after_date is not None:
            # Sort the files in order by date.
            files = sorted(files, key=DataExporter.get_date_from_file_name)
            # Remove files that are before this date.
            filtered_files = []
            for file_name in files:
                day = DataExporter.get_date_from_file_name(file_name)
                if day > load_after_date:
                    filtered_files.append(file_name)
            files = filtered_files
        return files

    @staticmethod
    def get_date_from_file_name(file_name):
        from datetime import datetime
        import pytz
        # file format: 2013_11_17_00_00_00.counts.json
        base_name = os.path.basename(file_name)
        base_name = base_name.split('.')[0] 
        return datetime.strptime(base_name, '%Y_%m_%d_%H_%M_%S').astimezone(pytz.UTC)

    @staticmethod
    def walkin(file_dir, files):
        import os
        list_file = os.listdir(file_dir)
        for file_name in list_file:
            file_path = os.path.join(file_dir, file_name)
            if os.path.isdir(file_path):
                DataExporter.walkin(file_path, files)
            else:
                if file_path.endswith('.json') or file_path.endswith('.gz'):
                    files.append(file_path)
