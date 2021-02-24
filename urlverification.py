import requests
import os
import pickle


class URLVerification:
    def __init__(self, top_level_domains):
        '''
        The init of this class iwll instantiate two dictionaries that contain the valid domain names and the defunct
        domain names respectively. This way, we can continuously track which URLs have already been seen. To make this
        run faster and faster, I will continually save this to a .pkl file.
        We also keep track of the top level domains here
        '''
        self.valid_urls = {}
        self.defunct_urls = {}
        self.top_level_domains = top_level_domains

    def extract_web_domain_from_url(self, url):
        '''
        Extract the actual website name from the URL, return it and the extension as a tuple of the website and its
        extension
        '''
        for d in self.top_level_domains:
            find_domain = url.partition(d)
            if len(find_domain[1]) > 0:
                # If the domain is not present in the URL, then the partition method will return the URL and
                # two empty strings. Thus, if the first term is greater than zero, we know we have found the
                # proper domain extension
                website = find_domain[0] + find_domain[1]
                extension = find_domain[2]
                return website, extension
        return "broken", "broken"

    '''
    Attempt to load the contents of the .pkl file into the valid/defunct urls dictionaries. 
    '''
    def load_valid_urls_from_pkl_file(self, valid_urls_filepath):
        if os.path.isfile(valid_urls_filepath):
            with open(valid_urls_filepath, 'rb') as f:
                self.valid_urls = pickle.load(f)

    def load_defunct_urls_from_pkl_file(self, defunct_urls_filepath):
        if os.path.isfile(defunct_urls_filepath):
            with open(defunct_urls_filepath, 'rb') as f:
                self.defunct_urls = pickle.load(f)

    def url_is_valid(self, url):
        '''
        This function will send a GET request to the domain provided, and return True if it can be hit and False if it
        cannot be
        '''
        website, extension = self.extract_web_domain_from_url(url)
        if website in self.valid_urls:
            # print(f"Website {website} was validated earlier")
            return True
        if website in self.defunct_urls:
            # print(f"Website {website} was marked as defunct earlier")
            return False
        try:
            # Bottleneck on performance is the very long requests, set a timeout. Probably should be shorter tbh
            r = requests.get(website, timeout=1)
            if website in self.valid_urls:
                self.valid_urls[website].append(extension)
            else:
                self.valid_urls[website] = [extension]
        except requests.exceptions.RequestException as e:
            # A whole lot of exceptions can be thrown here. If any one of them does, we can skip.
            # If a URL has already been verified as defunct, stop checking it again
            if website not in self.defunct_urls:
                # print(f"Website {website} is defunct because of {e}")
                self.defunct_urls[website] = True
            return False
        # print(f"Website {website} is valid")
        return True

