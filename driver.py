import json

import datacleaning as dc
import os
import spacy
from entityextraction import EntityExtraction
from urlverification import URLVerification
from graphconstructor import KnowledgeGraph
import pickle
import requests

if __name__ == '__main__':

    LENGTH_OF_GRAPHDB = 809780
    DICT_PICKLE_FILE_NAME = "knowledgegraphdict.pkl"
    URL_PICKLE_FILE_NAME = "knowledgegraphurls.pkl"
    VALID_URLS_PICKLE_FILE_NAME = "validurls.pkl"
    DEFUNCT_URLS_PICKLE_FILE_NAME = "defuncturls.pkl"
    FULL_SEEN_URLS_PICKLE_FILE_NAME = "fullseenurls.pkl"
    GRAPH_CONTENT_DF_PICKLE_FILE_NAME = "knowledgegraphdf.pkl"
    TEST_URLS_PICKLE_FILE_NAME = "testurls.pkl"
    column_labels = ['Entity', 'Relation', 'Value', 'Probability', 'Entity Literal Strings', 'Value Literal Strings',
                     'Best Entity Literal String', 'Best Value Literal String', 'Entity Categories', 'Value Categories']
    top_level_domains = [".com", ".edu", ".net", ".gov"]
    '''
    entities = dc.load_data_into_dict(LENGTH_OF_GRAPHDB, DICT_PICKLE_FILE_NAME)
    if not os.path.isfile(DICT_PICKLE_FILE_NAME):
        dc.to_pkl_file(DICT_PICKLE_FILE_NAME, entities)
    entities = {k: dc.delete_blank_entries_in_observation(v) for k, v in entities.items()
                if dc.validate_observation_structure(v)}
    '''
    urls_for_samples = dc.parse_urls_in_entities_dict(URL_PICKLE_FILE_NAME, LENGTH_OF_GRAPHDB)
    # Loading URLs and Graph dict takes total ~ 5 seconds
    if not os.path.isfile(URL_PICKLE_FILE_NAME):
        dc.to_pkl_file(URL_PICKLE_FILE_NAME, urls_for_samples)
    # Collect the extensions of the top level domains. We want to check if the websites are defunct or not first

    knowledge_graph_df = dc.entities_to_df(GRAPH_CONTENT_DF_PICKLE_FILE_NAME, column_labels)
    if not os.path.isfile(GRAPH_CONTENT_DF_PICKLE_FILE_NAME):
        dc.to_pkl_file(GRAPH_CONTENT_DF_PICKLE_FILE_NAME, knowledge_graph_df)

    # Load spaCy model
    nlp = spacy.load("en_core_web_sm")

    url_v = URLVerification(top_level_domains)
    # Attempt to load the valid/defunct URL dictionaries that we have saved to the .pkl file
    url_v.load_valid_urls_from_pkl_file(VALID_URLS_PICKLE_FILE_NAME)
    url_v.load_valid_urls_from_pkl_file(DEFUNCT_URLS_PICKLE_FILE_NAME)
    en = EntityExtraction(nlp)
    if os.path.isfile(TEST_URLS_PICKLE_FILE_NAME):
        with open(TEST_URLS_PICKLE_FILE_NAME, 'rb') as f:
            url_v.full_seen_urls = pickle.load(f)
    else:
        for key in urls_for_samples:
            url_list = urls_for_samples[key]
            for url in url_list:
                if len(url_v.full_seen_urls) == 20:
                    break
                if url_v.url_is_valid(url) and url not in url_v.full_seen_urls:
                    # Only proceed if the URL is not defunct
                    domain_terms = en.get_domain_terms_from_url(url)
                    if len(domain_terms) > 0:
                        print(url)
                        domain_terms = list(map(lambda x: x.strip(), domain_terms))
                        domain_terms = list(map(dc.remove_newline_delimiters, domain_terms))
                        domain_terms = list(map(dc.remove_tab_delimiters, domain_terms))
                        domain_terms = list(map(dc.transform_entities_to_match_graph_concept_format, domain_terms))
                        print(domain_terms)
                        url_v.full_seen_urls[url] = domain_terms
    if not os.path.isfile(TEST_URLS_PICKLE_FILE_NAME):
        dc.to_pkl_file(TEST_URLS_PICKLE_FILE_NAME, url_v.full_seen_urls)
    local_endpoint = "http://127.0.0.1:5000"
    local_endpoint += '/simmatrix'
    payload = {
        'url_list': json.dumps(url_v.full_seen_urls)
    }

    requests.post(local_endpoint, data=payload)
    knowledge_graph = KnowledgeGraph(knowledge_graph_df, urls_for_samples, url_v.full_seen_urls)
    knowledge_graph.construct_similarity_matrix()
    '''
    # Once we have finished our testing, save the verified defunct/valid URLs to a pkl file so we don't have to do it
    # over and over. NOTE: if you find a bug in the url verifier, DELETE THE .pkl FILES FOR THE VALID/DEFUNCT URLS; we
    # need to rebuild them and the pkl files have incorrect information in them
    dc.to_pkl_file(VALID_URLS_PICKLE_FILE_NAME, url_v.valid_urls)
    dc.to_pkl_file(DEFUNCT_URLS_PICKLE_FILE_NAME, url_v.defunct_urls)
    '''

