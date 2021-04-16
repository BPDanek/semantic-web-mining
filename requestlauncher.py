import requests
import argparse
import json

parser = argparse.ArgumentParser(description='Enter a comma separated list of URLs')
parser.add_argument('url_names', type=str, help='comma separated list of URLs to generate a similarity matrix for')
TEST_URLS_PICKLE_FILE_NAME = "testurls.pkl"
args = parser.parse_args()
url_list = args.url_names.split(",")
urls_with_terms = dict.fromkeys(url_list) # The values will be the domain terms associated with it.
local_endpoint = "http://127.0.0.1:5000"
local_endpoint += '/simmatrix'
# urls_with_terms = {k:v for k,v in urls_with_terms.items() if v is not None}
payload = {
    'url_list': json.dumps(url_list)
}
requests.post(local_endpoint, data=payload)
local_endpoint = "http://127.0.0.1:5000"
local_endpoint += '/getrecommendations'
payload = {
    'k': 2
}
requests.post(local_endpoint, data=payload)

