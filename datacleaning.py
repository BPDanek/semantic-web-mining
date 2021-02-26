import csv
import pickle
from collections import defaultdict
import time
import re
import os.path
import string
import urllib.parse
import pandas as pd


def remove_newline_delimiters(s):
    return re.sub('\x0a', '', s)


def remove_tab_delimiters(s):
    return re.sub('\x09', '', s)


def transform_entities_to_match_graph_concept_format(s):
    """
    Entities in the graph that consist of multiple words are connected with underscores i.e "Scott Cook" appears as
    scott_cook, so rewrite entries like that.
    """
    s = s.lower()
    # Remove punctuation, include one extra weird apostrophe that doesn't get caught normally
    s = s.translate(str.maketrans({c: '' for c in string.punctuation+'’'}))
    # Replace all spaces with underscores to connect terms consisting of multiple words
    s = re.sub(' ', '_', s)
    # Some extraneous spaces left after this, remove them
    return re.sub(' ', '', s)


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


def parse_urls_in_entities_dict(url_pkl_file_name, graph_db_length, entities=None):
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


def to_pkl_file(dict_file_name, entities):
    start = time.time()
    with open(dict_file_name, "wb") as f:
        pickle.dump(entities, f)
    print(f"Saved object to {dict_file_name} in {time.time() - start} seconds")
    f.close()


def validate_observation_structure(value):
    """
    This function will return true if the values in the array follow the schema presented, and false otherwise
    Runs quite fast tbh
    """
    try:
        # In some cases, the entity concept isn't actually a concept. Let's dump these observations
        entity_concept = value[-3].split()
        if not entity_concept[0].startswith("concept"):
            return False
    except IndexError:
        # Some entries are length 0. Drop these
        return False
    return True


def delete_blank_entries_in_observation(value):
    """
    Utility function that will remove blank entries from each observation. Also, drop the last entry in each of these
    lists (the source data) -  we have filtered the requisite information from these and no longer need them. We also
    want to replace all relations of type 'generalization' with
    """
    try:
        value.remove('')
        del value[-1]
    except ValueError:
        pass
    return value


def entities_to_df(graph_df_pickle_file_name, column_labels, entities=None):
    """
    Transform our entities dictionary into a Pandas Dataframe. Pass in the names of each of the columns, the entities
    dictionary itself, and the pickle file name to load if it is there
    """
    start = time.time()
    if os.path.isfile(graph_df_pickle_file_name):
        with open(graph_df_pickle_file_name, 'rb') as f:
            knowledge_graph_df = pickle.load(f)
        # The .pkl file loads in 2 seconds! Nice!
        print(f".pkl file loaded in {time.time() - start} seconds")
        return knowledge_graph_df
    else:
        knowledge_graph_df = pd.DataFrame.from_dict(entities, orient='index')
        # Last column is the URL information, we can dump it now
        knowledge_graph_df = knowledge_graph_df.drop([10], axis=1)
        # Label the columns of the dataframe
        knowledge_graph_df.columns = column_labels
        # We want to use the observation number as a key in the dictionary that holds all the URLs. Currently it is an
        # index so we make it a column
        knowledge_graph_df = knowledge_graph_df.reset_index()
        # Parse the entity literal strings into something that can actually be split
        knowledge_graph_df["Entity Literal Strings"] = knowledge_graph_df["Entity Literal Strings"].apply(
            lambda x: re.sub(r'"', '\x09', x) if x is not None else None)
        knowledge_graph_df["Value Literal Strings"] = knowledge_graph_df["Value Literal Strings"].apply(
            lambda x: re.sub(r'"', '\x09', x) if x is not None else None)
        # Some instances of extra underscores. Remove them here.
        knowledge_graph_df["Entity"] = knowledge_graph_df["Entity"].apply(
            lambda x: re.sub(r'(_)\1+', '_', str(x)) if x is not None else None)
        knowledge_graph_df["Value"] = knowledge_graph_df["Value"].apply(
            lambda x: re.sub(r'(_)\1+', '_', str(x)) if x is not None else None)
        # literal_strings_list = re.split(r"\t", literal_strings.iloc[0])
        # Only the odd indices carry what we want, so keep those
        # return [literal_strings_list[i] for i in range(len(literal_strings_list)) if i % 2 != 0]
    return knowledge_graph_df


