import numpy as np
import scipy.linalg as la
import pdb

from .submodular_funcs import *

class SubmodularOpt():

    def __init__(self, V=None, v=None, **kwargs):
        self.v = v
        self.V = V

    def initialize_function(self, lam, a1=1.0, a2=1.0, b1=1.0, b2= 1.0):
        self.a1 = a1
        self.a2 = a2
        self.b1 = b1
        self.b2 = b2

        self.noverlap_norm = ngram_overlap(self.v, self.V)
        self.ndistinct_norm = distinct_ngrams(self.V)
        self.sim_norm = similarity_func(self.v, self.V)
        self.edit_norm = np.sqrt(len(self.V))
        self.ndistinct_src_norm = distinct_src_ngrams_func(self.v,self.V) * 3
        self.lam = lam

    def final_func(self, pos_sets, rem_list, selec_set):
        distinct_score = np.array(list(map(distinct_ngrams, pos_sets)))/self.ndistinct_norm

        base_noverlap_score = ngram_overlap(self.v, selec_set)
        base_sim_score = similarity_func(self.v, selec_set)
        base_edit_score = seq_func(self.V, selec_set)
        base_distinct_src_score = distinct_src_ngrams_func(self.v, selec_set)

        noverlap_score = []
        for sent in rem_list:
            noverlap_score.append(ngram_overlap_unit(self.v, sent, base_noverlap_score))
        noverlap_score= np.array(noverlap_score)/self.noverlap_norm

        sim_score = []
        for sent in rem_list:
            sim_score.append(similarity_gain(self.v, sent, base_sim_score))
        sim_score= np.array(sim_score)/self.sim_norm

        ndistinct_score = []
        for sent in rem_list:
            ndistinct_score.append(distinct_src_ngrams_gain(self.v, sent, pos_sets, base_distinct_src_score))
        if self.ndistinct_src_norm == 0:
            self.ndistinct_src_norm = 1e9
        ndistinct_score = np.array(ndistinct_score)/self.ndistinct_src_norm

        edit_score = []
        for sent in rem_list:
            edit_score.append(seq_gain(self.v, sent, base_edit_score))
        edit_score= np.array(edit_score)/self.edit_norm

        quality_score = self.a1 * sim_score + self.a2 * noverlap_score
        diversity_score = self.b1 * ndistinct_score + self.b2 * edit_score

        final_score = self.lam * quality_score + (1-self.lam)* diversity_score

        return final_score

    def maximize_func(self, k=5):
        selec_sents= set()
        ground_set = set(self.V)
        selec_set = set(selec_sents)
        rem_set = ground_set.difference(selec_set)
        score = []
        while len(selec_sents) < k:

            rem_list = list(rem_set)
            pos_sets = [list(selec_set.union({x})) for x in rem_list]

            score_map = self.final_func(pos_sets, rem_list, selec_set)
            max_idx = np.argmax(score_map)
            score.append(score_map[max_idx])

            selec_sents = pos_sets[max_idx]
            selec_set= set(selec_sents)
            rem_set = ground_set.difference(selec_set)

        return selec_sents, score
