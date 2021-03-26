import numpy as np
from collections import Counter
import datacleaning as dc
from segmentation import segment


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
    def __init__(self, knowledge_graph_df, urls_for_samples, set_of_random_urls, nlp):
        self.knowledge_graph_df = knowledge_graph_df
        self.urls_for_samples = urls_for_samples
        # We may not need the following two attributes, but leave them in since they could be useful
        self.entity_concept_pairs = self.get_entity_types()
        # Get all unique types of entities
        self.entity_types = list(set(self.entity_concept_pairs.keys()))
        # Collect the unique values of the previous dictionary, representing all the unique concepts an
        # entity could take
        self.concept_types = list(set(self.entity_concept_pairs.values()))
        self.concept_types = list(map(self.segment_concept_names, self.concept_types))
        print(self.concept_types)
        # Track the number of times each unique concept appears, to be used for normalizing scores
        self.concept_counts = Counter(self.concept_types)
        self.relation_types = self.get_relation_types()
        # Collect a dictionary of URLs and their requisite domain terms. I want to try constructing a similarity matrix
        # based on the entity types and all that
        self.set_of_random_urls = set_of_random_urls
        self.nlp = nlp
        # Reindex starting from zero, since we are putting this information into a matrix

    def segment_concept_names(self, concept):
        # These concepts often consist of two words stitched together. Let's try to break them apart here
        concept_words = segment(concept)  # This adds a lot of overhead. We need to compute this earlier
        no_entries_length_1 = True
        for w in concept_words:
            # One seemingly effective way to see if the segmentation works is to see if there's
            # any entries in it of length 1. If so, use the original concept
            if len(w) == 1:
                no_entries_length_1 = False
                break
        if no_entries_length_1:
            # Case if segmentation was successful
            return " ".join(concept_words)
        else:
            return concept

    def get_relation_types(self):
        relation_types = self.knowledge_graph_df["Relation"].unique()
        relation_types = list(map(lambda x: x.partition("concept:"), relation_types))
        return dict.fromkeys(list(map(lambda x: x[2] if x[0] == '' else x[0], relation_types)))

    '''
    Create a list of pairs (entity_name, entity_type). For instance, (pfizer, biotechcompany) would be an output
    of this
    '''
    def get_entity_types(self):
        entity_types = self.knowledge_graph_df["Entity"].unique()
        # Keep only the terms that represent the "generalizations" relationship
        entity_type_tuples = list(map(lambda x: (x.split(":")[2], x.split(":")[1]) if len(x.split(":")) == 3
                                        else (x.split(":")[0], None), entity_types))
        # Return the samples that have the generalizations relationship AND skip the one case with a dash
        return {name: t for name, t in entity_type_tuples if t is not None and len(name) > 1}

    def construct_similarity_matrix(self):
        sim_matrix = np.zeros((len(self.set_of_random_urls), len(self.set_of_random_urls)))
        for k1, terms1 in self.set_of_random_urls.items():
            for k2, terms2 in self.set_of_random_urls.items():
                if k1 == k2:
                    continue
                common_terms = list(set(terms1).intersection(terms2)) # Find the terms present in both URLs
                if len(common_terms) > 0:
                    print(k1, k2, common_terms)
                sim_matrix[k1][k2] = len(common_terms) # The number of common elements in url i and url j
        # print(sim_matrix)

    '''
    Utility function to do some string content comparisons. This is for when there are multiple matches 
    present in the ReadTheWeb corpus and we need to check the distances between the matches, to get the most
    likely one.
    :param string1: first string
    :param string2: second string
    :return: the edit distance between the two
    '''
    def edit_distance(self, string1, string2):
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
        return memo[-1][-1]
    '''
    This function attempt to calculate the "concept" of a node that is not present in the ReadTheWeb ontology
    It does so by iterating over the entity-concept pairs in the DB, computing the similarity between the term and 
    the entity, and summing the score.
    
    '''
    def determine_concept_of_unknown_term(self, term):
        # First, check if the term is already present in the ReadTheWeb repository
        min_dist = float("inf")
        min_dist_match = ""
        # Change the term so it can be queried in the graph DB
        term_modified = dc.transform_entities_to_match_graph_concept_format(term)
        for entity, concept in self.entity_concept_pairs.items():
            if term_modified in entity:
                # print("Match found: ", entity, concept)
                if term_modified == entity:
                    # Terminate on exact match
                    return concept
                else:
                    dist = self.edit_distance(term.lower(), entity)
                    if dist < min_dist:
                        min_dist = dist
                        min_dist_match = concept
        if min_dist_match != "":
            # Only return if we happened to find a suitable match
            self.entity_concept_pairs.update({term_modified: min_dist_match})
            return min_dist_match
        similarity_scores = {c: 0 for c in self.concept_types}
        # Track the total similarity score for each of the unique entities
        # Compute the similarity between every entity we know of
        seen_concepts = {}
        threshold = 0
        for concept in self.concept_types:
            if concept in seen_concepts and similarity_scores[concept] < threshold:
                # print(concept, similarity_scores[concept])
                # If this concept has been encountered before and the similarity isn't high enough, skip
                continue
            tokens = self.nlp(concept + " " + term)
            try:
                if not tokens[-1].has_vector:
                    # Some words (like COVID-19) do not actually have a word vector yet. Thus, we
                    # return "unknown_concept" here since there's no way for us to know what they type is
                    print(f"No word vector for {term}")
                    return "unknown_concept"
                sim = tokens[0].similarity(tokens[-1])
                similarity_scores[concept] += sim
                # print([t.text for t in tokens], tokens[0].similarity(tokens[-1]))
            except KeyError:
                # Sometimes the tokenizer breaks the concept apart, so catch it here
                pass
            seen_concepts.update({concept: None})
        similarity_scores = {c: s/self.concept_counts[c] for c, s in similarity_scores.items()}
        return max(similarity_scores, key=similarity_scores.get)

