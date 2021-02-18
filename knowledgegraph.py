import csv
import pickle
from collections import defaultdict
import time
import re
import os.path
import urllib.parse

'''
Utility function that will replace all the tab separators in each list of URLs
'''
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
    return url

if __name__ == '__main__':
    LENGTH_OF_GRAPHDB = 1048576
    DICT_PICKLE_FILE_NAME = "knowledgegraphdict.pkl"
    start = time.time()
    entities = defaultdict()
    # Try to open the .pkl file, and also validate it by checking if it's the right length
    if os.path.isfile(DICT_PICKLE_FILE_NAME):
        with open(DICT_PICKLE_FILE_NAME, 'rb') as f:
            entities = pickle.load(f)
        if len(entities.keys()) != LENGTH_OF_GRAPHDB:
            print(f"Incorrect length: {len(entities.keys())} rows of {LENGTH_OF_GRAPHDB} present in .pkl file")
        # The .pkl file loads in 2 seconds! Nice!
        print(f"Finished in {time.time() - start} seconds")
    else:
        with open("knowledgegraphdata.csv", encoding="utf-8") as f:
            csv_reader = csv.reader(f, delimiter=",")
            next(csv_reader) # Skip the header
            for i, row in enumerate(csv_reader):
                # Split along the tab to get each entry in the CSV separated.
                record = row[0].split("\t")

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
        print(f"Finished in {time.time() - start} seconds")
        with open(DICT_PICKLE_FILE_NAME, "wb") as f:
            pickle.dump(entities, f)
    for value in entities.values():
        value[-1] = parse_urls(value[-1])
        print(value)


