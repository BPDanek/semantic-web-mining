import pandas as pd

import spacy


nlp = spacy.load('en_core_web_sm')

from spacy.matcher import Matcher

import networkx as nx

import matplotlib.pyplot as plt
from tqdm import tqdm

pd.set_option('display.max_colwidth', 200)


def get_entities(sent):
    ## chunk 1
    ent1 = ""
    ent2 = ""

    prv_tok_dep = ""  # dependency tag of previous token in the sentence
    prv_tok_text = ""  # previous token in the sentence

    prefix = ""
    modifier = ""

    #############################################################

    for tok in nlp(sent):
        ## chunk 2
        # if token is a punctuation mark then move on to the next token
        if tok.dep_ != "punct":
            # check: token is a compound word or not
            if tok.dep_ == "compound":
                prefix = tok.text
                # if the previous word was also a 'compound' then add the current word to it
                if prv_tok_dep == "compound":
                    prefix = prv_tok_text + " " + tok.text

            # check: token is a modifier or not
            if tok.dep_.endswith("mod") == True:
                modifier = tok.text
                # if the previous word was also a 'compound' then add the current word to it
                if prv_tok_dep == "compound":
                    modifier = prv_tok_text + " " + tok.text

            ## chunk 3
            if tok.dep_.find("subj") == True:
                ent1 = modifier + " " + prefix + " " + tok.text
                prefix = ""
                modifier = ""
                prv_tok_dep = ""
                prv_tok_text = ""

                ## chunk 4
            if tok.dep_.find("obj") == True:
                ent2 = modifier + " " + prefix + " " + tok.text

            ## chunk 5
            # update variables
            prv_tok_dep = tok.dep_
            prv_tok_text = tok.text
    #############################################################

    return [ent1.strip(), ent2.strip()]


def get_relation(sent):
    doc = nlp(sent)

    # Matcher class object
    matcher = Matcher(nlp.vocab)

    # define the pattern
    pattern = [{'DEP': 'ROOT'},
               {'DEP': 'prep', 'OP': "?"},
               {'DEP': 'agent', 'OP': "?"},
               {'POS': 'ADJ', 'OP': "?"}]

    matcher.add("matching_1", [pattern])

    matches = matcher(doc)
    k = len(matches) - 1

    span = doc[matches[k][1]:matches[k][2]]

    return span.text


candidate_sentences = pd.read_csv("/Users/midhungopalakrishnan/Desktop/test-data.csv")
candidate_sentences.shape
entity_pairs = []
relations = []

for i in tqdm(candidate_sentences["Search Data"]):
    entity_pairs.append(get_entities(i))
# print(entity_pairs[0:5])

for i in tqdm(candidate_sentences["Search Data"]):
    relations.append(get_relation(i))
# print(relations[0:5])

# extract subject
source = [i[0] for i in entity_pairs]

# extract object
target = [i[1] for i in entity_pairs]

# get distinct words list and mark relation between similar words
target_unique = []

for x in target:
    if x not in target_unique:
        target_unique.append(x)

for i in range(len(target_unique)-1): # 0 1 2....n-1
    for j in range(i+1, len(target_unique)):
        word1 = nlp(target_unique[i])
        word2 = nlp(target_unique[j])
        if word1.similarity(word2) > 0.70:
            print(target_unique[i], "similar to ", target_unique[j], " because of similarity ", word1.similarity(word2))
            source.append(target_unique[i])
            target.append(target_unique[j])
            relations.append("similar to")

kg_df = pd.DataFrame({'source': source, 'target': target, 'edge': relations})
# kg_df2 = pd.DataFrame({'source': wordList, 'target': wordList2, 'edge': similarityList})
# kg_df.append(kg_df2, ignore_index=True)

G = nx.from_pandas_edgelist(kg_df, "source", "target",
                            edge_attr=True, create_using=nx.MultiDiGraph())

plt.figure(figsize=(12, 12))

pos = nx.spring_layout(G)
nx.draw(G, with_labels=True, node_color='skyblue', edge_cmap=plt.cm.Blues, pos=pos)
plt.show()


