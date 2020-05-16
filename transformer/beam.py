""" Manage beam search info structure.
    Heavily borrowed from OpenNMT-py.
    For code in OpenNMT-py, please check the following link:
    https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/Beam.py
"""

import torch
import numpy as np
from data import data_utils
from transformer.submodopt import SubmodularOpt
from data.data_utils import convert_idx2text
import random


class Beam(object):
    ''' Store the necessary info for beam search '''
    def __init__(self, size, use_cuda=False):
        self.size = size
        self.done = False

        self.tt = torch.cuda if use_cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        #print('scores', self.scores.size())
        self.all_scores = []

        # The backpointers at each time-step.
        self.prev_ks = []

        # The outputs at each time-step.
        self.next_ys = [self.tt.LongTensor(size).fill_(data_utils.PAD)]
        #print(self.next_ys, len(self.next_ys))
        self.next_ys[0][0] = data_utils.BOS

    def get_current_state(self):
        "Get the outputs for the current timestep."
        return self.get_tentative_hypothesis()

    def get_current_origin(self):
        "Get the backpointers for the current timestep."
        return self.prev_ks[-1]

    def advance(self, word_lk):
        "Update the status and check for finished or not."
        num_words = word_lk.size(1)

        # Sum the previous scores.
        if len(self.prev_ks) > 0:
            beam_lk = word_lk + self.scores.unsqueeze(1).expand_as(word_lk)
        else:
            beam_lk = word_lk[0]
        # print("beam", beam_lk.size())
        flat_beam_lk = beam_lk.view(-1)
        # print(flat_beam_lk.size())
        # print("testtest",flat_beam_lk.size())

        best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True) # 1st sort
        best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True) # 2nd sort TODO

        # print("best_scores", best_scores, best_scores_id)
        self.all_scores.append(self.scores)
        self.scores = best_scores

        # bestScoresId is flattened beam_size * tgt_vocab_size array, so calculate
        # which word and beam each score came from
        prev_k = best_scores_id / num_words
        # print(prev_k)
        self.prev_ks.append(prev_k)
        # print(self.prev_ks[-1])
        self.next_ys.append(best_scores_id - prev_k * num_words)
        # print(best_scores_id, prev_k * num_words)
        # print(self.next_ys[-1])
        # print(self.next_ys)

        # End condition is when top-of-beam is EOS.
        if self.next_ys[-1][0] == data_utils.EOS:
            self.done = True
            self.all_scores.append(self.scores)

        return self.done

    def sort_scores(self):
        "Sort the scores."
        return torch.sort(self.scores, 0, True)

    def get_the_best_score_and_idx(self):
        "Get the score of the best in the beam."
        scores, ids = self.sort_scores()
        return scores[1], ids[1]

    def get_tentative_hypothesis(self):
        "Get the decoded sequence for the current timestep."

        if len(self.next_ys) == 1:
            dec_seq = self.next_ys[0].unsqueeze(1)
            # print("len_next_ys == 1")
            # print("len_next_ys == 1")
            # print("len_next_ys == 1")
            # print("len_next_ys == 1")
        else:
            _, keys = self.sort_scores()
            # print("keys", keys)
            hyps = [self.get_hypothesis(k) for k in keys]
            # print("hyps1", hyps)
            hyps = [[data_utils.BOS] + h for h in hyps]

            dec_seq = torch.from_numpy(np.array(hyps))

        return dec_seq

    def get_hypothesis(self, k):
        """
        Walk back to construct the full hypothesis.
        Parameters.
             * `k` - the position in the beam to construct.
         Returns.
            1. The hypothesis
            2. The attention at each time step.
        """
        hyp = []
        # print("self.next_ys", self.next_ys, len(self.next_ys))
        # print("self.prev_ks", self.prev_ks, len(self.prev_ks))
        for j in range(len(self.prev_ks)-1, -1, -1):
            
            hyp.append(int(self.next_ys[j + 1][k].item()))
            k = self.prev_ks[j][k]
        # print("final_word", temp_k, self.next_ys[len(self.prev_ks)][temp_k])
        # print(hyp)
        # print(hyp[::-1])

        return hyp[::-1]

    def advance_sub(self, word_lk, ori, tgt_sents):
        "Update the status and check for finished or not."
        #beam_size = word_lk.size(0)
        num_words = word_lk.size(1)

        # Sum the previous scores.
        if len(self.prev_ks) > 0:
            beam_lk = word_lk + self.scores.unsqueeze(1).expand_as(word_lk)
        else:
            beam_lk = word_lk[0]
        # print("beam", beam_lk.size())
        flat_beam_lk = beam_lk.view(-1)
        # print(flat_beam_lk.size())
        # print("testtest",flat_beam_lk.size())

        t_best_scores, t_best_scores_id = flat_beam_lk.topk(self.size, 0, True, True) # 1st sort
        t_best_scores, t_best_scores_id = flat_beam_lk.topk(self.size, 0, True, True) # 2nd sort TODO

        t_prev_k = t_best_scores_id / num_words
        t_next_ys = t_best_scores_id - t_prev_k * num_words


        _, _, tgt_idx2word = torch.load('./data/quora/pre.p.dict')['tgt_dict']
        #print("testtest", word_lk[0].size())
        candidate = []
        candidate_id = []
        # print(tgt_sents)
        for i in range(self.size):
            temp, tempid = word_lk[i].topk(2, 0, True, True)
            candidate.append(tempid)
            candidate_id.append([i]*2)
        #print(len(candidate),len(candidate[0]),candidate[0])
        #print(len(candidate_id), len(candidate_id[0]), candidate_id[0])
        new_tgt_sents = []
        m_state = {}
        count = 0
        for i in range(len(tgt_sents)):
            for word_idx in candidate[i]:
                idx_word = convert_idx2text([word_idx], tgt_idx2word)
                ttgt = tgt_sents[i] + " " + idx_word
                m_state[ttgt] = count
                new_tgt_sents.append(ttgt)
                count += 1
        #print(m_state)
        #print(len(new_tgt_sents))
        #print(new_tgt_sents[:20])
        subopt = SubmodularOpt(new_tgt_sents, ori)
        subopt.initialize_function(0.5,1,1,1,1)
        selec_sents, map_scores = subopt.maximize_func(self.size)
        #best_scores = torch.from_numpy(np.array(map_scores))
        # print(new_tgt_sents)
        # print(selec_sents)

        prev_k = []
        next_ys = []
        for i in range(3):
            prev_k.append(t_prev_k[i].item())
            next_ys.append(t_next_ys[i])

        for i in range(len(selec_sents)-3):
            idx = m_state[selec_sents[i]]
            ni = idx // 2
            wi = idx % 2
            prev_k.append(ni)
            next_ys.append(candidate[ni][wi])
        # print(selec_sents)
        # print(prev_k)
        # print(next_ys)

        self.all_scores.append(self.scores)
        self.scores = t_best_scores

        self.prev_ks.append(prev_k)
        self.next_ys.append(next_ys)

        # End condition is when top-of-beam is EOS.
        if self.next_ys[-1][0] == data_utils.EOS:
            self.done = True
            self.all_scores.append(self.scores)
        # for i in self.next_ys[-1]:
        #     if i == data_utils.EOS:
        #         self.done = True
        #         self.all_scores.append(self.scores)
        # if not self.done and len(self.next_ys) > len(ori):
        #     self.done = True
        #     self.all_scores.append(self.scores)

        return self.done

    def advance_sub_2(self, word_lk, ori, tgt_sents):
        "Update the status and check for finished or not."
        _, _, tgt_idx2word = torch.load('./data/quora/pre.p.dict')['tgt_dict']
        #print("testtest", word_lk[0].size())
        candidate = []
        candidate_id = []
        # print(tgt_sents)
        for i in range(self.size):
            temp, tempid = word_lk[i].topk(2, 0, True, True)
            candidate.append(tempid)
            candidate_id.append([i]*2)
        #print(len(candidate),len(candidate[0]),candidate[0])
        #print(len(candidate_id), len(candidate_id[0]), candidate_id[0])
        new_tgt_sents = []
        m_state = {}
        count = 0
        for i in range(len(tgt_sents)):
            for word_idx in candidate[i]:
                idx_word = convert_idx2text([word_idx], tgt_idx2word) if word_idx != data_utils.EOS else "<eos>"
                ttgt = tgt_sents[i] + " " + idx_word
                m_state[ttgt] = count
                new_tgt_sents.append(ttgt)
                count += 1
        #print(m_state)
        #print(len(new_tgt_sents))
        #print(new_tgt_sents[:20])
        ori = [ori[0] + " " + "<eos>"]
        subopt = SubmodularOpt(new_tgt_sents, ori)
        subopt.initialize_function(0.7,1,1,1,1)
        selec_sents, map_scores = subopt.maximize_func(self.size)
        best_scores = torch.from_numpy(np.array(map_scores))
        # print(new_tgt_sents)
        # print(selec_sents)

        prev_k = []
        next_ys = []

        for i in range(len(selec_sents)):
            idx = m_state[selec_sents[i]]
            ni = idx // 2
            wi = idx % 2
            prev_k.append(ni)
            next_ys.append(candidate[ni][wi])
        # print(selec_sents)
        # print(prev_k)
        # print(next_ys)

        self.all_scores.append(self.scores)
        self.scores = best_scores

        self.prev_ks.append(prev_k)
        self.next_ys.append(next_ys)

        # End condition is when top-of-beam is EOS.
        # if self.next_ys[-1][0] == data_utils.EOS:
        #     self.done = True
        #     self.all_scores.append(self.scores)
        for i in self.next_ys[-1]:
            if i == data_utils.EOS:
                self.done = True
                self.all_scores.append(self.scores)

        return self.done


    def advance_sub_3(self, word_lk, ori, tgt_sents, idx):
        "Update the status and check for finished or not."
        s = random.randint(0, 9)
        if idx+3 > len(ori[0]) or s >= 5:
            print("original")
            num_words = word_lk.size(1)
            # Sum the previous scores.
            if len(self.prev_ks) > 0:
                beam_lk = word_lk + self.scores.unsqueeze(1).expand_as(word_lk).cuda()
            else:
                beam_lk = word_lk[0]
            flat_beam_lk = beam_lk.view(-1)
            best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True) # 1st sort
            best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True) # 2nd sort TODO
            
            self.all_scores.append(self.scores)
            self.scores = best_scores
            
            # bestScoresId is flattened beam_size * tgt_vocab_size array, so calculate
            # which word and beam each score came from
            prev_k = best_scores_id / num_words
            self.prev_ks.append(prev_k)
            self.next_ys.append(best_scores_id - prev_k * num_words)
            
            # End condition is when top-of-beam is EOS.
            if self.next_ys[-1][0] == data_utils.EOS:
                self.done = True
                self.all_scores.append(self.scores)
        else:
            _, _, tgt_idx2word = torch.load('./data/quora/pre.p.dict')['tgt_dict']
            print("sub")
            candidate = []
            candidate_id = []
            for i in range(self.size):
                temp, tempid = word_lk[i].topk(2, 0, True, True)
                candidate.append(tempid)
                candidate_id.append([i]*2)
            new_tgt_sents = []
            m_state = {}
            count = 0
            for i in range(len(tgt_sents)):
                for word_idx in candidate[i]:
                    idx_word = convert_idx2text([word_idx], tgt_idx2word)
                    ttgt = tgt_sents[i] + " " + idx_word
                    m_state[ttgt] = count
                    new_tgt_sents.append(ttgt)
                    count += 1
            subopt = SubmodularOpt(new_tgt_sents, ori)
            subopt.initialize_function(0.7,1,1,1,1)
            selec_sents, map_scores = subopt.maximize_func(self.size)
            best_scores = torch.from_numpy(np.array(map_scores))
            
            prev_k = []
            next_ys = []
            
            for i in range(len(selec_sents)):
                idx = m_state[selec_sents[i]]
                ni = idx // 2
                wi = idx % 2
                prev_k.append(ni)
                next_ys.append(candidate[ni][wi])
            self.all_scores.append(self.scores)
            self.scores = best_scores
            self.prev_ks.append(prev_k)
            self.next_ys.append(next_ys)
            # End condition is when top-of-beam is EOS.
            if self.next_ys[-1][0] == data_utils.EOS:
                self.done = True
                self.all_scores.append(self.scores)

        return self.done
