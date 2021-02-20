import datacleaning as dc
import os
from bs4 import BeautifulSoup
import requests
from collections import defaultdict

if __name__ == '__main__':
    LENGTH_OF_GRAPHDB = 809780
    DICT_PICKLE_FILE_NAME = "knowledgegraphdict.pkl"
    URL_PICKLE_FILE_NAME = "knowledgegraphurls.pkl"
    entities = dc.load_data_into_dict(LENGTH_OF_GRAPHDB, DICT_PICKLE_FILE_NAME)
    if not os.path.isfile(DICT_PICKLE_FILE_NAME):
        dc.dict_to_pkl_file(DICT_PICKLE_FILE_NAME, entities)
    entities_1 = dc.parse_urls_in_entities_dict(URL_PICKLE_FILE_NAME, entities, LENGTH_OF_GRAPHDB)
    # Loading URLs and Graph dict takes total ~ 5 seconds
    if not os.path.isfile(URL_PICKLE_FILE_NAME):
        dc.dict_to_pkl_file(URL_PICKLE_FILE_NAME, entities_1)
    # Collect the extensions of the top level domains. We want to check if the websites are defunct or not first
    top_level_domains = [".com", ".edu", ".net", ".gov"]
    valid_urls = defaultdict()
    for key, value in entities_1.items():
        for url in value:
            try:
                for d in top_level_domains:
                    find_domain = url.partition(d)
                    if len(find_domain[1]) > 0:
                        # If the domain is not present in the URL, then the partition method will return the URL and
                        # two empty strings. Thus, if the first term is greater than zero, we know we have found the
                        # proper domain extension
                        website = find_domain[0] + find_domain[1]
                        extension = find_domain[2]
                        r = requests.get(website)
                        # If the request works, we can place this URL into a dictionary as a key
                        if valid_urls[website]:
                            valid_urls[website].append(extension)
                        else:
                            valid_urls[website] = [extension]
            except Exception:
                print(f"The URL {url} is defunct")
            break
        print("\n")