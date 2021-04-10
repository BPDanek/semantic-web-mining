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
    LENGTH_OF_GRAPHDB = 798252
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
    entities = dc.load_data_into_dict(LENGTH_OF_GRAPHDB, DICT_PICKLE_FILE_NAME)
    dc.to_pkl_file(DICT_PICKLE_FILE_NAME, entities)
    urls_for_samples = dc.parse_urls_in_entities_dict(URL_PICKLE_FILE_NAME, LENGTH_OF_GRAPHDB, entities)
    # Loading URLs and Graph dict takes total ~ 5 seconds
    if not os.path.isfile(URL_PICKLE_FILE_NAME):
        dc.to_pkl_file(URL_PICKLE_FILE_NAME, urls_for_samples)
    # Collect the extensions of the top level domains. We want to check if the websites are defunct or not first

    knowledge_graph_df = dc.entities_to_df(GRAPH_CONTENT_DF_PICKLE_FILE_NAME, column_labels, entities)
    if not os.path.isfile(GRAPH_CONTENT_DF_PICKLE_FILE_NAME):
        dc.to_pkl_file(GRAPH_CONTENT_DF_PICKLE_FILE_NAME, knowledge_graph_df)
    nlp = spacy.load("en_core_web_lg")
    knowledge_graph = KnowledgeGraph(knowledge_graph_df, nlp)
    # Load spaCy model


