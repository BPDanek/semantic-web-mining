from collections import Counter
from bs4 import BeautifulSoup
import requests
import re


'''
spaCy entity types and what they mean

CARDINAL         Numerals that do not fall under another type
DATE     Absolute or relative dates or periods
EVENT    Named hurricanes, battles, wars, sports events, etc.
FAC      Buildings, airports, highways, bridges, etc.
GPE      Countries, cities, states
LANGUAGE         Any named language
LAW      Named documents made into laws.
LOC      Non-GPE locations, mountain ranges, bodies of water
MONEY    Monetary values, including unit
NORP     Nationalities or religious or political groups
ORDINAL          "first", "second", etc.
ORG      Companies, agencies, institutions, etc.
PERCENT          Percentage, including "%"
PERSON   People, including fictional
PRODUCT          Objects, vehicles, foods, etc. (not services)
QUANTITY         Measurements, as of weight or distance
TIME     Times smaller than a day
WORK_OF_ART      Titles of books, songs, etc.

'''


class EntityExtraction:
    def __init__(self, nlp):
        # Pass spaCy object as parameter
        self.nlp_model = nlp

    def collect_metadata_tags(self, file_html):
        '''
        On many web documents, the <meta> tags can contain useful information about the topic of the article. However,
        not every website has them. If it doesn't we return a blank list
        '''
        soup = BeautifulSoup(file_html, 'html.parser')
        # Locate all the <meta> tags in a document that have the name=keyword moniker somewhere. This gives us a list of
        # topics that the article is about.
        meta_tag = soup.find("meta", {"name": re.compile(".*keyword*")})

        tags = []
        try:
            # Split the string on both the commas and the semicolons i.e. Coronavirus;COVID-19 is a single term in the
            # tags but should be separated
            tags = re.split('[,;]', meta_tag["content"])
            if len(tags) == 1 and tags[0] == "null":
                # Some NYT articles have this meta tag but no terms in it, query the <p> tags
                tags = self.get_text_from_paragraph_tags(file_html)
        except (KeyError, TypeError):
            # Some files do not have keyword tags. Try to find the description and title, concatenate them, and
            # perform entity extraction on that
            # If even this fails, we just scrape the <p> tags. I'd rather not do this
            tags = self.get_text_from_paragraph_tags(file_html)
        return tags

    def get_text_from_paragraph_tags(self, file_html):
        '''
        If the description and metadata don't tell us enough, we can put the text of this article through spaCy as well
        This is a super fallback; <p> tag concatentation introduces a lot of garbage into our text and we don't want
        to do it unless we have to.
        '''
        soup = BeautifulSoup(file_html, 'html.parser')
        paragraph_tags = soup.findAll("p")
        text = []
        for node in paragraph_tags:
            text += ''.join(node.findAll(text=True))
        return ''.join(text)

    def get_domain_terms_from_url(self, url):
        '''
        Function that will return a list of domain terms for each URL provided. An empty list means that the URL does not
        provide a response anymore and should be discarded.
        Pass in the NLP model as a parameter so we only have to create it once
        '''
        # Set the headers for our GET requests so that it looks like the requests are coming from a browser
        # and not a Python script
        domain_terms = []
        headers = {
            "User-Agent": 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/51.0.2704.103 Safari/537.36',
            'Content-Type': 'text/html; charset=utf-8',
        }
        # Collect the html from the page and create a parser for it
        try:
            r = requests.get(url, headers=headers, timeout=1)
        except Exception:
            # A whole lot of exceptions can be thrown here. If any one of them does, we can skip.
            # If a URL has already been verified as defunct, stop checking it again
            print("Request failed")
            return domain_terms
        file_html = r.text
        response_code = r.status_code
        if response_code != 200:
            # In the case of a failed connection, return a blank list to signal that the URL is defunct.
            print("Request failed!")
            return []
        metadata_tags = self.collect_metadata_tags(file_html)
        if isinstance(metadata_tags, str):
            # If this function returns a string, we are dealing with an og:description tag and want to put it through
            # the spaCy entity extraction
            try:
                description = self.nlp_model(metadata_tags)
            except ValueError:
                # The limit on the number of characters that can be used in a spaCy model is 1 million, so trim the text
                # down to that many characters and recompute the description
                description = self.nlp_model(metadata_tags[:1000000])

            # Only track some of the entities, some, like numbers and percentages, are useless. Still deliberating on
            # using NORP in this...will see later
            useful_entity_types = ['EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 'ORG', 'PERSON', 'PRODUCT',
                                   'WORK_OF_ART']
            # Checking for keys in dict is constant time vs linear time for checking for element in list, this will make
            # collecting the actual entities a lot faster
            useful_entity_types = {k: True for k in useful_entity_types}
            # Collect the useful entities that will appear in our knowlegde graph
            domain_terms = [ent.text for ent in description.ents if ent.label_ in useful_entity_types]
        else:
            # If the tags are not of string type, they are a list of tags extracted from the metadata. We can return
            # these immediately
            domain_terms = metadata_tags
        # Sometimes duplicates emerge in the set of entities. Why? Who the hell knows. Let's select the 15 most common
        # in each and return them as domain terms
        return [x for x, y in Counter(domain_terms).most_common(10)]

    def extract_web_domain_from_url(self, top_level_domains, url):
        '''
        Extract the actual website name from the URL, return it and the extension as a tuple of the website and its
        extension
        '''
        for d in top_level_domains:
            find_domain = url.partition(d)
            if len(find_domain[1]) > 0:
                # If the domain is not present in the URL, then the partition method will return the URL and
                # two empty strings. Thus, if the first term is greater than zero, we know we have found the
                # proper domain extension
                website = find_domain[0] + find_domain[1]
                extension = find_domain[2]
                return website, extension
        return "", ""

