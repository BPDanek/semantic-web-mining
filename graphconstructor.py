import numpy as np
from collections import Counter
import datacleaning as dc
import pandas as pd
from segmentation import segment
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer
pd.options.mode.chained_assignment = None  # default='warn'


class KnowledgeGraph:
    '''
    :param knowledge_graph_df: a DataFrame containing the rows of observations and the requisite information
                                that comes with each one; i.e the type of relation it partakes in, the URLs that
                                this info was gleaned from, etc.
    :param urls_for_samples: Dict where keys are enumerated sample #s and values are the set of URLs that defined the
                                relationship in that sample
    :param set_of_random_urls: A subset of the urls_for_samples parameter, just in for ease of testing right now. Will
                                remove later
    :param nlp: spaCy object
    '''
    def __init__(self, knowledge_graph_df, urls_for_samples, nlp):
        self.knowledge_graph_df = knowledge_graph_df
        self.urls_for_samples = urls_for_samples
        self.nlp = nlp
        self.segmentation_mappings = {}
        self.edit_distances = {} # Track the edit distances so we don't recompute
        self.triples = self.generate_triples()
        # When segmenting, don't check segmentations twice. Track the seen ones here
        # Gather the unique entities, relations, and values
        self.entity_types = self.triples["Entity"].unique()
        self.relation_types = self.triples["Relation"].unique()
        self.concept_types = self.triples["Segmented Concept"].unique()
        # Track the number of times each unique concept appears, to be used for normalizing scores

        self.concept_counts = Counter(self.triples["Segmented Concept"])
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
            return " ".join(concept_words)
        else:
            # print(concept)
            self.segmentation_mappings.update({concept: concept})
            return concept

    def get_relation_types(self):
        relation_types = self.knowledge_graph_df["Relation"].unique()
        relation_types = list(map(lambda x: x.partition("concept:"), relation_types))
        return dict.fromkeys(list(map(lambda x: x[2] if x[0] == '' else x[0], relation_types)))

    '''
    :param url_set: Dict where key = index and value = the URL it enumerates
    '''
    def construct_similarity_matrix(self, url_set):
        sim_matrix = np.zeros((len(url_set), len(url_set)))
        for k1, terms1 in url_set.items():
            for k2, terms2 in url_set.items():
                if k1 == k2:
                    continue
                common_terms = list(set(terms1).intersection(terms2)) # Find the terms present in both URLs
                if len(common_terms) > 0:
                    print(k1, k2, common_terms)
                sim_matrix[k1][k2] = len(common_terms) # The number of common elements in url i and url j
        # print(sim_matrix)

    '''
    Directly match terms from the sets of two domain terms and return their similarity via a float
    '''
    def direct_domain_term_matches(self, domain_terms1, domain_terms2):
        # Want to avoid missing matches between the same word with different case
        domain_terms1 = list(map(dc.transform_entities_to_match_graph_concept_format, domain_terms1))
        domain_terms2 = list(map(dc.transform_entities_to_match_graph_concept_format, domain_terms2))
        domain_terms1, domain_terms2 = self.remove_duplicates(domain_terms1, domain_terms2)
        print(domain_terms1)
        print(domain_terms2)
        full_set_of_domain_terms = list(set(domain_terms1 + domain_terms2))
        print(full_set_of_domain_terms)
        # Transform to dict keys to allow for easier lookup
        domain_terms1 = dict.fromkeys(domain_terms1)
        domain_terms2 = dict.fromkeys(domain_terms2)
        vector1 = list(map(lambda x: 1 if x in domain_terms1 else 0, full_set_of_domain_terms))
        vector2 = list(map(lambda x: 1 if x in domain_terms2 else 0, full_set_of_domain_terms))
        print("Vector 1: ", vector1)
        print("Vector 2: ", vector2)
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
        if candidates.shape[0] != 0:
            # If a match is found, we can return
            entities_to_check = candidates.loc[:, 'Entity']
            candidates.loc[:, "NELL Match Sim"] = entities_to_check.apply(self.edit_distance, string2=term_modified)
            # Return the candidate with the cheapest edit distance (most semantically similar)
            # print(candidates.loc[candidates["NELL Match Sim"].idxmin()]["Entity"])
            print("Direct match found: " + candidates.loc[candidates["NELL Match Sim"].idxmin()]["Entity"])
            return candidates.loc[candidates["NELL Match Sim"].idxmin()]["Segmented Concept"]
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
                        sim = max(sim, sim1)
                    except KeyError:
                        # Sometimes the tokenizer breaks the concept apart, so catch it here
                        continue
            return sim
        concept_similarities = pd.DataFrame(self.concept_types, columns=["Concept"])
        concept_similarities["Scores"] = pd.Series(self.concept_types).apply(get_semantic_similarity, term=term)
        concept_similarities = concept_similarities.sort_values(by=["Scores"], ascending=False)
        # Scale by the number
        # concept_similarities["Scores"] = concept_similarities["Scores"].apply(lambda x: x/self.concept_counts[x])
        concept_similarities.to_csv("concepts.csv")
        self.edit_distances.clear() # Edit distances only relevant to this specific term, wipe this dict
        # Return concept associated
        return concept_similarities.loc[concept_similarities["Scores"].idxmax()]["Concept"]


