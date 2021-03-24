import requests
import argparse
import os
import json
import spacy
from entityextraction import EntityExtraction
import datacleaning as dc
import pickle

parser = argparse.ArgumentParser(description='Enter a filename containing urls to be loaded (preferably a .txt file)')
parser.add_argument('filepath', type=str, help='Name of the file containing the URLs. Make sure it is present in this '
                                               'directory!')
TEST_URLS_PICKLE_FILE_NAME = "testurls.pkl"
args = parser.parse_args()
filepath = args.filepath
filepath = os.path.join(os.getcwd(), filepath)
urls_with_terms = {}
url_list = []
with open(filepath, 'r') as f:
    for line in f:
        url_list.append(line.strip())
    f.close()
urls_with_terms = dict.fromkeys(url_list) # The values will be the domain terms associated with it.
nlp = spacy.load("en_core_web_sm")
en = EntityExtraction(nlp)
for url in urls_with_terms:
    domain_terms = en.get_domain_terms_from_url(url)  # Extract domain terms from the urls
    if len(domain_terms) > 0:
        domain_terms = list(map(lambda x: x.strip(), domain_terms))
        domain_terms = list(map(dc.remove_newline_delimiters, domain_terms))
        domain_terms = list(map(dc.remove_tab_delimiters, domain_terms))
        domain_terms = list(map(dc.transform_entities_to_match_graph_concept_format, domain_terms))
        urls_with_terms[url] = domain_terms
local_endpoint = "http://127.0.0.1:5000"
local_endpoint += '/simmatrix'
urls_with_terms = {k:v for k,v in urls_with_terms.items() if v is not None}
payload = {
    'url_list': json.dumps(urls_with_terms)
}
requests.post(local_endpoint, data=payload)
