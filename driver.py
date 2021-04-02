import json

import datacleaning as dc
import os
import spacy
from entityextraction import EntityExtraction
from urlverification import URLVerification
from graphconstructor import KnowledgeGraph
import pickle
import time

if __name__ == '__main__':
    LENGTH_OF_GRAPHDB = 809780
    DICT_PICKLE_FILE_NAME = "knowledgegraphdict.pkl"
    URL_PICKLE_FILE_NAME = "knowledgegraphurls.pkl"
    VALID_URLS_PICKLE_FILE_NAME = "validurls.pkl"
    DEFUNCT_URLS_PICKLE_FILE_NAME = "defuncturls.pkl"
    FULL_SEEN_URLS_PICKLE_FILE_NAME = "fullseenurls.pkl"
    GRAPH_CONTENT_DF_PICKLE_FILE_NAME = "knowledgegraphdf.pkl"
    KNOWLEDGE_GRAPH_PICKLE_FILE_NAME = "fullknowledgegraph.pkl"
    SEGMENTATIONS_PICKLE_FILE_NAME = "segmentations.pkl"
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
    nlp = spacy.load("en_core_web_lg")
    url_v = URLVerification(top_level_domains)
    # Attempt to load the valid/defunct URL dictionaries that we have saved to the .pkl file
    url_v.load_valid_urls_from_pkl_file(VALID_URLS_PICKLE_FILE_NAME)
    url_v.load_valid_urls_from_pkl_file(DEFUNCT_URLS_PICKLE_FILE_NAME)
    en = EntityExtraction(nlp)
    '''
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
                    print(url)
                    if len(domain_terms) > 0:
                        domain_terms = list(map(lambda x: x.strip(), domain_terms))
                        domain_terms = list(map(dc.remove_newline_delimiters, domain_terms))
                        domain_terms = list(map(dc.remove_tab_delimiters, domain_terms))
                        domain_terms = list(map(dc.transform_entities_to_match_graph_concept_format, domain_terms))
                        print(domain_terms)
                        url_v.full_seen_urls[url] = domain_terms
    '''
    if not os.path.isfile(TEST_URLS_PICKLE_FILE_NAME):
        dc.to_pkl_file(TEST_URLS_PICKLE_FILE_NAME, url_v.full_seen_urls)
    url_list = {i: url_v.full_seen_urls[key] for i, key in enumerate(list(url_v.full_seen_urls.keys()))}
    if os.path.isfile(KNOWLEDGE_GRAPH_PICKLE_FILE_NAME):
        with open(KNOWLEDGE_GRAPH_PICKLE_FILE_NAME, 'rb') as f:
            knowledge_graph = pickle.load(f)
    else:
        knowledge_graph = KnowledgeGraph(knowledge_graph_df, urls_for_samples, nlp)
    # dc.to_pkl_file(KNOWLEDGE_GRAPH_PICKLE_FILE_NAME, knowledge_graph)
    # Save every time, to capture changes we make
    url = 'https://www.theguardian.com/us-news/2021/apr/02/mlb-baseball-all-star-game-georgia-voting-law'
    start = time.time()
    domain_terms = en.get_domain_terms_from_url(url)
    # print(domain_terms)
    concept_list = []
    '''
        for term in domain_terms:
        concept = knowledge_graph.determine_concept_of_unknown_term(term)
        if concept != "unknown_concept":
            # Track the list of concepts
            concept_list.append(concept)
        print(f"Most likely concept for {term}: " + concept)
    print(time.time() - start)
    '''

    url = 'https://www.nytimes.com/2021/04/02/us/politics/mlb-all-star-game-moved.html?action=click&module=' \
          'Top%20Stories&pgtype=Homepage'
    # start = time.time()
    domain_terms_1 = en.get_domain_terms_from_url(url)
    # print(domain_terms_1)
    # concept_list = []
    '''
        for term in domain_terms:
        concept = knowledge_graph.determine_concept_of_unknown_term(term)
        if concept != "unknown_concept":
            # Track the list of concepts
            concept_list.append(concept)
        print(f"Most likely concept for {term}: " + concept)
    print(time.time() - start)
    '''

    print(knowledge_graph.direct_domain_term_matches(domain_terms, domain_terms_1))
    # knowledge_graph.construct_similarity_matrix()
    '''
    # Once we have finished our testing, save the verified defunct/valid URLs to a pkl file so we don't have to do it
    # over and over. NOTE: if you find a bug in the url verifier, DELETE THE .pkl FILES FOR THE VALID/DEFUNCT URLS; we
    # need to rebuild them and the pkl files have incorrect information in them
    dc.to_pkl_file(VALID_URLS_PICKLE_FILE_NAME, url_v.valid_urls)
    dc.to_pkl_file(DEFUNCT_URLS_PICKLE_FILE_NAME, url_v.defunct_urls)
    '''

