'''
NOTE: I ripped this from Peter Norvig, from this link:
https://stackoverflow.com/questions/38125281/split-sentence-without-space-in-python-nltk
THIS IS NOT MY ORIGINAL WORK

'''
from __future__ import division
import nltk
from collections import Counter

WORDS = nltk.corpus.brown.words()
COUNTS = Counter(WORDS)


def pdist(counter):
    "Make a probability distribution, given evidence from a Counter."
    N = sum(counter.values())
    return lambda x: counter[x]/N


P = pdist(COUNTS)


def Pwords(words):
    "Probability of words, assuming each word is independent of others."
    return product(P(w) for w in words)


def product(nums):
    "Multiply the numbers together.  (Like `sum`, but with multiplication.)"
    result = 1
    for x in nums:
        result *= x
    return result


def splits(text, start=0, L=20):
    "Return a list of all (first, rest) pairs; start <= len(first) <= L."
    return [(text[:i], text[i:])
            for i in range(start, min(len(text), L)+1)]


def segment(text):
    "Return a list of words that is the most probable segmentation of text."
    if not text:
        return []
    else:
        candidates = ([first] + segment(rest)
                      for (first, rest) in splits(text, 1))
        return max(candidates, key=Pwords)