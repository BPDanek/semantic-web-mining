from flask import Flask
from flask import request
from graphconstructor import KnowledgeGraph
import datacleaning as dc
from urlverification import URLVerification
import os
import json

app = Flask(__name__)


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
urls_for_samples = dc.parse_urls_in_entities_dict(URL_PICKLE_FILE_NAME, LENGTH_OF_GRAPHDB)
# Loading URLs and Graph dict takes total ~ 5 seconds
if not os.path.isfile(URL_PICKLE_FILE_NAME):
    dc.to_pkl_file(URL_PICKLE_FILE_NAME, urls_for_samples)
url_v = URLVerification(top_level_domains)
# Attempt to load the valid/defunct URL dictionaries that we have saved to the .pkl file
url_v.load_valid_urls_from_pkl_file(VALID_URLS_PICKLE_FILE_NAME)
url_v.load_valid_urls_from_pkl_file(DEFUNCT_URLS_PICKLE_FILE_NAME)
'''
This method will take input as a series of URLs and return a similarity matrix, where
entry (i,j) is the number of common domain terms between URL i and URL j
'''


@app.route('/simmatrix', methods=['POST'])
def return_sim_matrix():
    url_list = json.loads(request.form['url_list']) # Collect the list of URLs
    # Convert the various columns in the ReadTheWeb CSV into a DF to use them more easily
    url_list = {i: url_list[key] for i, key in enumerate(list(url_list.keys()))}
    knowledge_graph_df = dc.entities_to_df(GRAPH_CONTENT_DF_PICKLE_FILE_NAME, column_labels)
    if not os.path.isfile(GRAPH_CONTENT_DF_PICKLE_FILE_NAME):
        dc.to_pkl_file(GRAPH_CONTENT_DF_PICKLE_FILE_NAME, knowledge_graph_df)
    knowledge_graph = KnowledgeGraph(knowledge_graph_df, urls_for_samples, url_list)
    knowledge_graph.construct_similarity_matrix()
    return url_list
