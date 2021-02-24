import datacleaning as dc
import os
import spacy
import entityextraction as en
from urlverification import URLVerification

if __name__ == '__main__':

    LENGTH_OF_GRAPHDB = 809780
    DICT_PICKLE_FILE_NAME = "knowledgegraphdict.pkl"
    URL_PICKLE_FILE_NAME = "knowledgegraphurls.pkl"
    VALID_URLS_PICKLE_FILE_NAME = "validurls.pkl"
    DEFUNCT_URLS_PICKLE_FILE_NAME = "defuncturls.pkl"

    entities = dc.load_data_into_dict(LENGTH_OF_GRAPHDB, DICT_PICKLE_FILE_NAME)
    if not os.path.isfile(DICT_PICKLE_FILE_NAME):
        dc.dict_to_pkl_file(DICT_PICKLE_FILE_NAME, entities)
    urls_for_samples = dc.parse_urls_in_entities_dict(URL_PICKLE_FILE_NAME, entities, LENGTH_OF_GRAPHDB)
    # Loading URLs and Graph dict takes total ~ 5 seconds
    if not os.path.isfile(URL_PICKLE_FILE_NAME):
        dc.dict_to_pkl_file(URL_PICKLE_FILE_NAME, urls_for_samples)
    # Collect the extensions of the top level domains. We want to check if the websites are defunct or not first
    top_level_domains = [".com", ".edu", ".net", ".gov"]
    '''
    valid_urls = html_info.check_for_defunct_urls(entities_1, top_level_domains, VALID_URLS_PICKLE_FILE_NAME)
    if os.path.isfile(VALID_URLS_PICKLE_FILE_NAME):
        dc.dict_to_pkl_file(VALID_URLS_PICKLE_FILE_NAME, valid_urls)
    '''
    # Best Value Literal Strings is usually empty, and the source column exists solely for the creation/validation
    # of URLs for the graph.
    column_labels = ['Entity', 'Relation', 'Value', 'Probability', 'Entity Literal Strings',
                     'Best Value Literal Strings', 'Entity Categories', 'Value Categories', 'Source']
    entities = {k: dc.delete_blank_entries_in_observation(v) for k, v in entities.items()
                if dc.validate_observation_structure(v)}
    # Load spaCy model
    nlp = spacy.load("en_core_web_sm")
    url_v = URLVerification(top_level_domains)
    # Attempt to load the valid/defunct URL dictionaries that we have saved to the .pkl file
    url_v.load_valid_urls_from_pkl_file(VALID_URLS_PICKLE_FILE_NAME)
    url_v.load_valid_urls_from_pkl_file(DEFUNCT_URLS_PICKLE_FILE_NAME)
    for key in list(entities.keys())[50:500]:
        obs = entities[key]
        entity = obs[0].split(":")[-1]
        relation = obs[1]
        value = obs[2].split(":")[-1]
        url_list = urls_for_samples[key]
        for url in url_list:
            if url_v.url_is_valid(url):
                # Only proceed if the URL is not defunct
                domain_terms = en.get_domain_terms_from_url(url, nlp)
                if len(domain_terms) > 0:
                    print(url)
                    # Strip out the following: 1) whitespace on either side, 2) newline characters and 3) tab delimiters
                    domain_terms = list(map(lambda x: x.strip(), domain_terms))
                    domain_terms = list(map(dc.remove_newline_delimiters, domain_terms))
                    domain_terms = list(map(dc.remove_tab_delimiters, domain_terms))
                    print(domain_terms)
    # Once we have finished our testing, save the verified defunct/valid URLs to a pkl file so we don't have to do it
    # over and over. NOTE: if you find a bug in the url verifier, DELETE THE .pkl FILES FOR THE VALID/DEFUNCT URLS; we
    # need to rebuild them and the pkl files have incorrect information in them
    dc.dict_to_pkl_file(VALID_URLS_PICKLE_FILE_NAME, url_v.valid_urls)
    dc.dict_to_pkl_file(DEFUNCT_URLS_PICKLE_FILE_NAME, url_v.defunct_urls)
