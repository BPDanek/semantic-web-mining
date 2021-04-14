import numpy as np
from collections import Counter
import datacleaning as dc
import pandas as pd
from segmentation import segment
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer
import pickle
from entityextraction import EntityExtraction
import os
pd.options.mode.chained_assignment = None  # default='warn'


class KnowledgeGraph:
    '''
    :param knowledge_graph_df: a DataFrame containing the rows of observations and the requisite information
                                that comes with each one; i.e the type of relation it partakes in, the URLs that
                                this info was gleaned from, etc.
    :param nlp: spaCy object
    '''
    def __init__(self, knowledge_graph_df, nlp):
        self.pkl_file_name = "segmentedgraphtriples.pkl"
        self.sim_matrix_pkl_file_name = "simmatrix.pkl"
        self.url_indices_mappings_pkl_file_name = "urlindices.pkl"
        self.url_domain_terms_pkl_file_name = "urldomainterms.pkl"
        self.url_concepts_pkl_file_name = "urlconceptsterms.pkl"
        self.knowledge_graph_df = knowledge_graph_df
        self.nlp = nlp
        self.entity_extraction = EntityExtraction(nlp)
        self.segmentation_mappings = {}
        self.segmentation_mappings_inverted = {}
        if not os.path.isfile(self.url_indices_mappings_pkl_file_name):
            self.indices_for_urls = {}
        else:
            with open(self.url_indices_mappings_pkl_file_name, 'rb') as f:
                self.indices_for_urls = pickle.load(f)
        self.c = 0.6
        self.edit_distances = {} # Track the edit distances so we don't recompute
        if not os.path.isfile(self.sim_matrix_pkl_file_name):
            self.sim_matrix = None
        else:
            with open(self.sim_matrix_pkl_file_name, 'rb') as f:
                self.sim_matrix = pickle.load(f)
        # Track the domain terms seen for a URL so we don't have to recalculate them. Saves computation time
        if not os.path.isfile(self.url_domain_terms_pkl_file_name):
            self.url_domain_terms = {}
        else:
            with open(self.url_domain_terms_pkl_file_name, 'rb') as f:
                self.url_domain_terms = pickle.load(f)

        # Same for the concepts
        if not os.path.isfile(self.url_concepts_pkl_file_name):
            self.url_concepts = {}
        else:
            with open(self.url_concepts_pkl_file_name, 'rb') as f:
                self.url_concepts = pickle.load(f)
        if not os.path.isfile(self.pkl_file_name):
            self.triples = self.generate_triples()
            dc.to_pkl_file(self.pkl_file_name, self.triples)
        else:
            with open(self.pkl_file_name, 'rb') as f:
                self.triples = pickle.load(f)
        # When segmenting, don't check segmentations twice. Track the seen ones here
        # Gather the unique entities, relations, and values
        self.entity_types = self.triples["Entity"].unique()
        self.relation_types = self.triples["Relation"].unique()
        self.concept_types = self.triples["Segmented Concept"].unique()
        # Normalize the number of times a concept appears so we can scale each concept by its frequency. Higher
        # frequency concepts (like personus, or country) should be weighted down, since two concepts labeled "personus"
        # are not quite related
        # Collect a dictionary of URLs and their requisite domain terms. I want to try constructing a similarity matrix
        # based on the entity types and all that
        # Reindex starting from zero, since we are putting this information into a matrix

    '''
    Create a set of (entity, relation, Concept) triples from the first three columns of the dataframe
    '''
    def generate_triples(self):
        triples_df = pd.DataFrame(columns=["Entity", "Relation", "Concept", "Segmented Concept"])
        # Extract the name of the entity
        self.knowledge_graph_df = self.knowledge_graph_df.loc[
            ~self.knowledge_graph_df["Value"].str.contains('\d+\.*\d+')]
        triples_df["Entity"] = self.knowledge_graph_df["Entity"].apply(lambda x: x.split(":")[-1])
        triples_df["Relation"] = self.knowledge_graph_df["Relation"].apply(lambda x: x.split(":")[-1])
        # Splitting on concept to catch the 'haswikipediaurl' case where value is a URL and : split fails
        triples_df["Concept"] = self.knowledge_graph_df["Value"].apply(lambda x: x.split(":")[1]
                                                                    if not x.startswith("http") and
                                                                       len(x.split(":")) > 1 else x)
        triples_df["Concept Counts"] = triples_df.groupby("Concept")["Concept"].transform('count')
        triples_df["Value Literal Strings"] = self.knowledge_graph_df["Value Literal Strings"].apply(
            lambda x: x.split("\t") if x is not None else [])
        triples_df["Entity Literal Strings"] = self.knowledge_graph_df["Entity Literal Strings"].apply(
            lambda x: x.split("\t") if x is not None else [])
        # triples_df.to_csv("triples.csv")
        # print("Saved file")
        # Partition the names of the concepts, to save computation in the actual graph construction
        triples_df["Segmented Concept"] = triples_df["Concept"].apply(self.segment_concept_names)

        return triples_df

    def segment_concept_names(self, concept):
        # These concepts often consist of two words stitched together. Let's try to break them apart here
        if concept.startswith("http"):
            # Occasionally a URL will appear
            return concept
        if '_' in concept:
            return concept.replace('_', ' ')
        if concept in self.segmentation_mappings:
            # If we've segmented this normally, return
            return self.segmentation_mappings[concept]
        concept_words = segment(concept)  # This adds a lot of overhead. We need to compute this earlier
        no_entries_length_1 = True
        for w in concept_words:
            # One seemingly effective way to see if the segmentation works is to see if there's
            # any entries in it of length 1. If so, use the original concept
            if len(w) == 1:
                no_entries_length_1 = False
                break
        if no_entries_length_1:
            # Case if segmentation was
            # print(" ".join(concept_words))
            self.segmentation_mappings.update({concept: " ".join(concept_words)})
            # Need to map back from segmentations to the original in a later step
            self.segmentation_mappings_inverted.update({" ".join(concept_words): concept})
            return " ".join(concept_words)
        else:
            # print(concept)
            self.segmentation_mappings.update({concept: concept})
            self.segmentation_mappings_inverted.update({concept: concept})
            return concept

    def get_relation_types(self):
        relation_types = self.knowledge_graph_df["Relation"].unique()
        relation_types = list(map(lambda x: x.partition("concept:"), relation_types))
        return dict.fromkeys(list(map(lambda x: x[2] if x[0] == '' else x[0], relation_types)))

    '''
    Given two URLs, this function will return a float value between 0 and 1 capturing their semantic similarity
    '''
    def get_similarity_of_two_urls(self, url1, url2):
        if url1 in self.url_domain_terms:
            domain_terms = self.url_domain_terms[url1]
        else:
            domain_terms = self.entity_extraction.get_domain_terms_from_url(url1)
            self.url_domain_terms.update({url1: domain_terms})
        print(url1, domain_terms)
        if url1 in self.url_concepts:
            # print(f"{url1} concepts found already", self.url_concepts[url1])
            concept_list = self.url_concepts[url1]
        else:
            concept_list = []
            for term in domain_terms:
                concept = self.determine_concept_of_unknown_term(term)
                if concept != "unknown_concept":
                    # Track the list of concepts
                    concept_list.append(concept)
                # print(f"Most likely concept for {term}: " + concept)
            self.url_concepts.update({url1: concept_list})
        if url2 in self.url_domain_terms:
            domain_terms_1 = self.url_domain_terms[url2]
        else:
            domain_terms_1 = self.entity_extraction.get_domain_terms_from_url(url2)
            self.url_domain_terms.update({url2: domain_terms_1})
        print(url2, domain_terms_1)
        if url2 in self.url_concepts:
            # print(f"{url2} concepts found already", self.url_concepts[url2])
            concept_list_1 = self.url_concepts[url2]
        else:
            concept_list_1 = []
            for term in domain_terms_1:
                concept = self.determine_concept_of_unknown_term(term)
                if concept != "unknown_concept":
                    # Track the list of concepts
                    concept_list_1.append(concept)
                # print(f"Most likely concept for {term}: " + concept)
            self.url_concepts.update({url2: concept_list_1})
        domain_term_matching_score = self.direct_domain_term_matches(domain_terms, domain_terms_1)[0][0]
        # print("Domain term matching", domain_term_matching_score)
        concept_matching_score = self.direct_domain_term_matches(concept_list, concept_list_1)[0][0]
        # print("Concept matching score", concept_matching_score)
        print("Total similarity", self.harmonic_mean(domain_term_matching_score, concept_matching_score))
        return self.harmonic_mean(domain_term_matching_score, concept_matching_score)

    '''
    Executes pagerank on the similarity matrix
    ASSUMPTIONS: Some URLs have already been collected => a similarity matrix exists
    :param k: Domain terms from top k URLs 
    '''
    def pagerank(self, k):
        i = 1
        v = np.full(self.sim_matrix.shape[0], 1/self.sim_matrix.shape[0])
        u = v
        while i < 25:
            u_new = ((1-self.c) * np.dot(self.sim_matrix, u)) + (self.c * v)
            u = u_new
            i += 1
        # Get the URLs that best capture the user's browsing history, in descending order to
        # get the most dominant ones first
        top_urls = np.argsort(-u)
        # Collect a list of domain terms that best capture the user's browser history
        best_domain_terms = []
        for index in top_urls[:k]:
            url = self.indices_for_urls[index]
            best_domain_terms += self.url_domain_terms[url]
        print(top_urls)
        print(list(set(best_domain_terms)))
        dc.to_pkl_file(self.url_concepts_pkl_file_name, self.url_concepts)
        dc.to_pkl_file(self.url_domain_terms_pkl_file_name, self.url_domain_terms)
        return list(set(best_domain_terms))

    '''
    :param url_set: List of the URLs 
    '''
    def construct_similarity_matrix(self, url_list):
        # Same URLs should have a similarity of one
        if self.sim_matrix is None:
            print("No sim matrix exists, building a new one")
            sim_matrix = np.zeros((len(url_list), len(url_list)))
            # Line up the indices of the matrix with the URL that they correspond to
            self.indices_for_urls = {i: url for i, url in enumerate(url_list)}
            np.fill_diagonal(sim_matrix, 1)
            for i in range(len(url_list)):
                for j in range(i + 1, len(url_list)):
                    # Only check top half, since matrix is symmetric
                    url1, url2 = url_list[i], url_list[j]
                    if url1 == url2:
                        # Perhaps two URLs end up being the same. Catch it here
                        sim_matrix[i][j] = 1
                        continue
                    sim_matrix[i][j] = self.get_similarity_of_two_urls(url1, url2)
            # Now, populate the bottom half of the matrix
            for i in range(len(url_list)):
                for j in range(i + 1, len(url_list)):
                    sim_matrix[j][i] = sim_matrix[i][j]
            print(sim_matrix)
            self.sim_matrix = sim_matrix
            self.save_sim_matrix_to_pkl_file()
            self.save_indices_to_pkl_file()
            return sim_matrix
        else:
            print("Sim matrix exists, adding to the one that's already there")
            # If the matrix exists already, we want to add the new URL domain terms to the existing ones
            sim_matrix = self.sim_matrix
            print(sim_matrix.shape)
            starting_index = len(self.indices_for_urls)
            # Only select the URLs that haven't been seen before
            url_list = [u for u in url_list if u not in self.url_domain_terms]
            for i in range (starting_index, starting_index + len(url_list)):
                # Already have a list of URLs, so we need to add to the original ones
                self.indices_for_urls.update({i: url_list[i-starting_index]})
            # Reshape the similarity matrix to account for the new URL
            zero_row = np.zeros((len(url_list), self.sim_matrix.shape[0]))
            sim_matrix = np.concatenate((sim_matrix, zero_row), axis=0)
            zero_col = np.zeros((len(url_list), sim_matrix.shape[0]))
            sim_matrix = np.concatenate((sim_matrix, zero_col.T), axis=1)
            print(sim_matrix.shape)
            # Leverage the fact that the majority of the sim matrix has already been calculated
            for i in range(sim_matrix.shape[1]):
                # Fill out all the new columns
                for j in range(sim_matrix.shape[0] - len(url_list), sim_matrix.shape[0]):
                    print(i,j)

                    # Only check top half, since matrix is symmetric
                    url1, url2 = url_list[j-sim_matrix.shape[0]], self.indices_for_urls[i]
                    if url1 == url2 or i == j:
                        # Perhaps two URLs end up being the same. Catch it here
                        sim_matrix[i][j] = 1
                        continue
                    sim_matrix[i][j] = self.get_similarity_of_two_urls(url1, url2)
            for i in range(sim_matrix.shape[0]):
                for j in range(sim_matrix.shape[1]):
                    sim_matrix[j][i] = sim_matrix[i][j]
            self.sim_matrix = sim_matrix
            # Keep it in the pkl file to access later
            self.save_sim_matrix_to_pkl_file()
            self.save_indices_to_pkl_file()
            return sim_matrix

    def harmonic_mean(self, a, b):
        return (a+b)/2

    '''
    Directly match terms from the sets of two domain terms and return their similarity via a float
    '''
    def direct_domain_term_matches(self, domain_terms1, domain_terms2):
        # Want to avoid missing matches between the same word with different case
        domain_terms1 = list(map(dc.transform_entities_to_match_graph_concept_format, domain_terms1))
        domain_terms2 = list(map(dc.transform_entities_to_match_graph_concept_format, domain_terms2))
        domain_terms1, domain_terms2 = self.remove_duplicates(domain_terms1, domain_terms2)
        # print(domain_terms1)
        # print(domain_terms2)
        full_set_of_domain_terms = list(set(domain_terms1 + domain_terms2))
        # print(full_set_of_domain_terms)
        # Transform to dict keys to allow for easier lookup
        domain_terms1 = dict.fromkeys(domain_terms1)
        domain_terms2 = dict.fromkeys(domain_terms2)

        vector1 = list(map(lambda x: 1 if x in domain_terms1 else 0, full_set_of_domain_terms))
        vector2 = list(map(lambda x: 1 if x in domain_terms2 else 0, full_set_of_domain_terms))
        # print("Vector 1: ", vector1)
        # print("Vector 2: ", vector2)
        return cosine_similarity(np.array([vector1]), np.array([vector2]))

    '''
    Possible that some terms are repeated but not exact matches e.g. Voting and Voters and 
    some terms contained entirely. We want to first stem the words w/ PorterStemmer and 
    then do some matching
    '''
    def remove_duplicates(self, terms1, terms2):
        ps = PorterStemmer()
        terms1 = list(map(lambda x: ps.stem(x), terms1))
        terms2 = list(map(lambda x: ps.stem(x), terms2))
        for i in range(len(terms1)):
            for j in range(len(terms2)):
                t, t1 = terms1[i], terms2[j]
                if t == t1:
                    continue
                elif t in t1:
                    # If the word t is contained in t1, then it replace t1 because they mean the same thing
                    # and it shrinks the joint list
                    terms1[i] = terms2[j]
                elif t1 in t:
                    terms2[j] = terms1[i]
        return terms1, terms2

    '''
    Simple utility function that saves the dataframe of relevant graph data to a pkl file. Kept here
    to simplify the syntax in driver
    '''
    def save_triples_df_to_pkl_file(self):
        dc.to_pkl_file(self.pkl_file_name, self.triples)

    def save_sim_matrix_to_pkl_file(self):
        dc.to_pkl_file(self.sim_matrix_pkl_file_name, self.sim_matrix)

    def save_indices_to_pkl_file(self):
        dc.to_pkl_file(self.url_indices_mappings_pkl_file_name, self.indices_for_urls)
    '''
    Utility function to do some string content comparisons. This is for when there are multiple matches 
    present in the ReadTheWeb corpus and we need to check the distances between the matches, to get the most
    likely one.
    :param string1: first string
    :param string2: second string
    :return: the edit distance between the two
    '''
    def edit_distance(self, string1, string2):
        if string1 in self.edit_distances:
            return self.edit_distances[string1]
        memo = np.zeros((len(string1) + 1, len(string2) + 1))
        for i in range(len(memo)):
            memo[i][0] = i
        for i in range(len(memo[0])):
            memo[0][i] = i
        for i in range(1, len(memo)):
            for j in range(1, len(memo[0])):
                if string1[i - 1] == string2[j - 1]:
                    memo[i][j] = memo[i - 1][j - 1]
                else:
                    memo[i][j] = 1 + max(memo[i - 1][j], memo[i][j - 1], memo[i - 1][j - 1])
        self.edit_distances[string1] = memo[-1][-1]
        return memo[-1][-1]
    '''
    This function attempt to calculate the "concept" of a node that is not present in the ReadTheWeb ontology
    It does so by iterating over the entity-concept pairs in the DB, computing the similarity between the term and 
    the entity, and summing the score.
    '''
    def determine_concept_of_unknown_term(self, term):
        # First, check if the term is already present in the ReadTheWeb repository
        # Change the term so it can be queried in the graph DB
        term_modified = dc.transform_entities_to_match_graph_concept_format(term)
        candidates = self.triples.loc[(self.triples["Relation"] == "generalizations") &
                                      (self.triples["Entity"].str.contains(term_modified))]
        # candidates.to_csv("candidates.csv")
        if candidates.shape[0] != 0:
            entities_to_check = candidates.loc[:, 'Entity']
            candidates.loc[:, "NELL Match Sim"] = entities_to_check.apply(self.edit_distance, string2=term_modified)
            # Return the candidate with the cheapest edit distance (most semantically similar)
            # print(candidates.loc[candidates["NELL Match Sim"].idxmin()]["Entity"])
            if candidates.loc[candidates["NELL Match Sim"].idxmin()]["NELL Match Sim"] == 0:
                # Return direct matches here. I know it would be smarter to simply change the condition to
                # build candidates, but it doesn't work for some reason
                # print("Direct match found: " + candidates.loc[candidates["NELL Match Sim"].idxmin()]["Entity"])
                return candidates.loc[candidates["NELL Match Sim"].idxmin()]["Segmented Concept"]
            candidates["Matches"] = candidates["Entity Literal Strings"].apply(
                lambda x: True if term_modified in x else False)
            # First, check the set of value literal strings (similar strings to the ones we've isolated), and if there
            #  is a direct match, return
            if candidates["Matches"].any():
                # If we find a match in the set of aliases, return that
                direct_matches = candidates.loc[candidates["Matches"]]["Segmented Concept"].reset_index(drop=True)
                return direct_matches.loc[0]
        # Track the total similarity score for each of the unique entities
        # Compute the similarity between every entity we know of

        def get_semantic_similarity(concept, term):
            if concept.startswith("http"):
                return -10
            token_length = len(term.split(" "))
            concept_length = len(concept.split(" "))
            tokens = self.nlp(concept + " " + term)
            sim = 0
            for i in range(concept_length):
                for j in range(concept_length, concept_length + token_length):
                    if not tokens[i].has_vector or not tokens[j].has_vector:
                        # Some words (like COVID-19) do not actually have a word vector yet. Thus, we
                        # return "unknown_concept" here since there's no way for us to know what they type is
                        # print("No word vector for one of the terms")
                        continue
                    try:
                        sim1 = tokens[i].similarity(tokens[j])
                        # print(tokens[i], tokens[j], sim)
                        sim += (sim1**2)
                    except KeyError:
                        # Sometimes the tokenizer breaks the concept apart, so catch it here
                        continue
            return sim

        concept_similarities = pd.DataFrame(self.concept_types, columns=["Concept"])
        concept_similarities["Scores"] = pd.Series(self.concept_types).apply(get_semantic_similarity, term=term)
        # Lots of columns where the concepts were URLs, drop those
        concept_similarities = concept_similarities.loc[concept_similarities["Scores"] >= 0]
        if (concept_similarities["Scores"] == 0).all():
            # A full slate of zeros means that we don't have a word vector for this concept -> this concept is unknown
            # If there were matches found in NELL that weren't direct, we fall back to those
            if candidates.shape[0] != 0:
                # print("Closest match", candidates.loc[candidates["NELL Match Sim"].idxmin()]["Entity"])
                return candidates.loc[candidates["NELL Match Sim"].idxmin()]["Segmented Concept"]
            # print("Unknown concept", term)
            return "unknown_concept"
        concept_similarities = concept_similarities.sort_values(by=["Scores"], ascending=False)
        self.edit_distances.clear() # Edit distances only relevant to this specific term, wipe this dict
        best_concept = concept_similarities.loc[concept_similarities["Scores"].idxmax()]["Concept"]
        # Add the newly delineated concept to the corpus, to allow for easier search later on
        best_concept_unsegmented = self.triples.loc[self.triples["Segmented Concept"] ==
                                                    best_concept]["Concept"].iloc[0]
        new_row = pd.DataFrame([[term_modified, "generalizations", best_concept_unsegmented, best_concept, 1, "", ""]],
                               columns=["Entity", "Relation", "Concept", "Segmented Concept", "Concept Counts",
                                        "Value Literal Strings", "Entity Literal Strings"])
        self.triples = self.triples.append(new_row, ignore_index=True)
        # print(self.triples.shape)
        # Return concept associated
        return best_concept


