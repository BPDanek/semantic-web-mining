import numpy as np


class KnowledgeGraph:
    def __init__(self, knowledge_graph_df, urls_for_samples, set_of_random_urls):
        self.knowledge_graph_df = knowledge_graph_df
        self.urls_for_samples = urls_for_samples
        # We may not need the following two attributes, but leave them in since they could be useful
        self.entity_types = self.get_entity_types()
        self.relation_types = self.get_relation_types()
        self.entities = {}
        self.relations = {}
        # Collect a dictionary of URLs and their requisite domain terms. I want to try constructing a similarity matrix
        # based on the entity types and all that
        self.set_of_random_urls = set_of_random_urls
        # Reindex starting from zero, since we are putting this information into a matrix
        self.set_of_random_urls = {i: v for i, v in enumerate(set_of_random_urls.values())}

    def get_relation_types(self):
        relation_types = self.knowledge_graph_df["Relation"].unique()
        relation_types = list(map(lambda x: x.partition("concept:"), relation_types))
        return dict.fromkeys(list(map(lambda x: x[2] if x[0] == '' else x[0], relation_types)))

    def get_entity_types(self):
        entity_types = self.knowledge_graph_df["Entity"].unique()
        # Create a list of pairs (entity_name, entity_type). For instance, (pfizer, biotechcompany) would be an output
        # of this
        entity_type_tuples = list(map(lambda x: (x.split(":")[2], x.split(":")[1])
                                                if len(x.split(":")) == 3 else (x.split(":")[0], None), entity_types))
        return {name: t for name, t in entity_type_tuples if t is not None}

    def construct_similarity_matrix(self):
        sim_matrix = np.zeros((len(self.set_of_random_urls), len(self.set_of_random_urls)))
        for k1, terms1 in self.set_of_random_urls.items():
            for k2, terms2 in self.set_of_random_urls.items():
                if k1 == k2:
                    continue
                common_terms = list(set(terms1).intersection(terms2))
                if len(common_terms) > 0:
                    print(k1, k2, common_terms)
                sim_matrix[k1][k2] = len(common_terms)
        print(sim_matrix)


class WebPage:
    def __init__(self, url, domain_terms):
        self.url = url
        self.domain_terms = domain_terms
        self.concepts = {}

