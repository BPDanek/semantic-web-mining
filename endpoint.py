from flask import Flask
from flask import request
from graphconstructor import KnowledgeGraph
import datacleaning as dc
import spacy
import numpy as np
import json

app = Flask(__name__)

GRAPH_CONTENT_DF_PICKLE_FILE_NAME = "knowledgegraphdf.pkl"
column_labels = ['Entity', 'Relation', 'Value', 'Probability', 'Entity Literal Strings', 'Value Literal Strings',
                 'Best Entity Literal String', 'Best Value Literal String', 'Entity Categories', 'Value Categories']
nlp = spacy.load("en_core_web_lg")
knowledge_graph_df = dc.entities_to_df(GRAPH_CONTENT_DF_PICKLE_FILE_NAME, column_labels)
knowledge_graph = KnowledgeGraph(knowledge_graph_df, nlp)


@app.route('/simmatrix', methods=['POST'])
def return_sim_matrix():
    url_list = json.loads(request.form['url_list']) # Collect the list of URLs
    # Convert the various columns in the ReadTheWeb CSV into a DF to use them more easily
    sim_matrix = knowledge_graph.construct_similarity_matrix(url_list)
    knowledge_graph.save_triples_df_to_pkl_file()
    return json.dumps(sim_matrix.tolist())
