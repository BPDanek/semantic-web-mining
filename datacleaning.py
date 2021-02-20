import csv
import pickle
from collections import defaultdict
import time
import re
import os.path
import urllib.parse


def remove_tab_delimiters(s):
    return re.sub('\x09', '', s)


def parse_urls(s):
    # The first part of the last element of the record is metadata, but http can be used to isolate the URL
    # which could come in handy later
    url = ''.join(s.partition("http")[1:])
    # The %09 character is giving me some serious trouble when parsing URLs. According to HTML docs, it is
    # a horizontal tab, and I will replace it with a + to see if it fixes the URLs
    url = re.sub("%09", "+", url)
    # Since the URLs are stored as strings recovered from the html pages of these websites. We want to
    # replace all those characters with the real character
    url = urllib.parse.unquote(url).split('+')
    # Map a function that remove tab characters left over from urllib conversion. Now, we have a list of
    # fully functional URLs
    url = list(map(remove_tab_delimiters, url))
    # Strip out all non-urls
    # Map + filter takes ~ 10 minutes, but list comp takes around 2.5 minutes
    url = [x for x in url if x.startswith("http")]
    # Deletion of non-url elements results in a whole bunch of None entries, filter them out
    return url


def load_data_into_dict(graph_db_length, dict_pickle_file_name):
    start = time.time()
    entities = defaultdict()
    # Try to open the .pkl file, and also validate it by checking if it's the right length
    if os.path.isfile(dict_pickle_file_name):
        with open(dict_pickle_file_name, 'rb') as f:
            entities = pickle.load(f)
        if len(entities.keys()) != graph_db_length:
            print(f"Incorrect length: {len(entities.keys())} rows of {graph_db_length} present in .pkl file")
        # The .pkl file loads in 2 seconds! Nice!
        print(f".pkl file loaded in {time.time() - start} seconds")
        return entities
    else:
        with open("knowledgegraphdata.csv", encoding="utf-8") as f:
            csv_reader = csv.reader(f, delimiter=",")
            next(csv_reader)  # Skip the header
            for i, row in enumerate(csv_reader):
                # Split along the tab to get each entry in the CSV separated.
                record = row[0].split("\t")
                # Sometimes the entire record cannot fit on the first line of the csv and carries over to the next. To
                # address this, we concatenate all the subsequent lines until we get to a line that begins with the
                # "concept" substring, indicating the end of the entry
                # This separation also leaves a string of length 0 as an entry each time, so try to strip it out here
                try:
                    record.remove('')
                except ValueError:
                    pass
                # Entries 3 and 5 are just metadata from the actual training of the NELL model,
                # so I dump those here. I keep the actual confidence score since that determines the weight of the
                # edge in the knowledge graph
                try:
                    del record[3]
                    del record[4]
                except IndexError:
                    pass
                # For some reason, some records end up length 0, possibly due to malformed URLs and metadata in the file
                # If so, dump them here. We also want to dump records for which the confidence is below 0.95; I found
                # an instance in which Alan Ogg, a basketball player, was marked as a bacterium because he died of such
                # a disease, and the threshold was 0.95. This is probably arbitrary, but will hold for now
                try:
                    if len(record) == 0 or float(record[3]) <= 0.95:
                        continue
                except IndexError:
                    pass
                entities[i] = record
            # Loads entire knowledge graph in 54 seconds...how much faster is pkl?
        print(f"Finished loading into dict in {time.time() - start} seconds, length is {len(entities.keys())}")
        return entities


def parse_urls_in_entities_dict(url_pkl_file_name, entities, graph_db_length):
    start = time.time()
    if os.path.isfile(url_pkl_file_name):
        with open(url_pkl_file_name, 'rb') as f:
            entities_1 = pickle.load(f)
        if len(entities_1.keys()) != graph_db_length:
            print(f"Incorrect length: {len(entities.keys())} rows of {graph_db_length} present in .pkl file")
        # The .pkl file loads in 2 seconds! Nice!
        print(f".pkl file loaded in {time.time() - start} seconds")
        return entities_1
    entities_1 = dict(map(lambda x: (x[0], parse_urls(x[1][-1])), entities.items()))
    # Dict/zip method takes 246.497 seconds
    # Map method takes 218.458 seconds
    print(f"Finished cleaning URLs in {time.time() - start} seconds, length is {len(entities_1.keys())}")
    return entities_1


def dict_to_pkl_file(dict_file_name, entities):
    start = time.time()
    with open(dict_file_name, "wb") as f:
        pickle.dump(entities, f)
    print(f"Saved object to {dict_file_name} in {time.time() - start} seconds")
    f.close()




