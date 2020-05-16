# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable

from transformer.models import Transformer
from transformer.beam import Beam
from transformer.submodopt import SubmodularOpt
from data.data_utils import convert_idx2text


class Translator(object):
    ''' Load with trained model and handel the beam search '''
    def __init__(self, opt, use_cuda):
        self.opt = opt
        self.use_cuda = use_cuda
        self.tt = torch.cuda if use_cuda else torch

        checkpoint = torch.load(opt.model_path)
        model_opt = checkpoint['opt']

        self.model_opt = model_opt
        model = Transformer(model_opt)
        if use_cuda:
            print('Using GPU..')
            model = model.cuda()

        prob_proj = nn.LogSoftmax(dim=-1)
        model.load_state_dict(checkpoint['model_params'])
        print('Loaded pre-trained model_state..')

        self.model = model
        self.model.prob_proj = prob_proj
        self.model.eval()

    def translate_batch(self, src_batch):
        ''' Translation work in one batch '''

        # Batch size is in different location depending on data.
        enc_inputs, enc_inputs_len = src_batch
        batch_size = enc_inputs.size(0) # enc_inputs: [batch_size x src_len]
        beam_size = self.opt.beam_size

        # Encode
        enc_outputs, _ =  self.model.encode(enc_inputs, enc_inputs_len)
        # Repeat data for beam
        enc_inputs = enc_inputs.repeat(1, beam_size).view(batch_size * beam_size, -1)
        enc_outputs =  enc_outputs.repeat(1, beam_size, 1).view(
            batch_size * beam_size, enc_outputs.size(1), enc_outputs.size(2))

        # Prepare beams
        beams = [Beam(beam_size, self.use_cuda) for _ in range(batch_size)]
        beam_inst_idx_map = {
            beam_idx: inst_idx for inst_idx, beam_idx in enumerate(range(batch_size))
        }
        print("beam_inst_idx_map",beam_inst_idx_map)
        n_remaining_sents = batch_size


        _, _, tgt_idx2word = torch.load('./data/quora/pre.p.dict')['tgt_dict']
        # Decode
        for i in range(self.opt.max_decode_step):
            len_dec_seq = i + 1
            # Preparing decoded data_seq
            # size: [batch_size x beam_size x seq_len]
            dec_partial_inputs = torch.stack([
                b.get_current_state() for b in beams if not b.done])
            
            # size: [batch_size * beam_size x seq_len]
            dec_partial_inputs = dec_partial_inputs.view(-1, len_dec_seq)
            # wrap into a Variable
            dec_partial_inputs = Variable(dec_partial_inputs, volatile=True)

            # Preparing decoded pos_seq
            # size: [1 x seq]
            # dec_partial_pos = torch.arange(1, len_dec_seq + 1).unsqueeze(0) # TODO:
            # # size: [batch_size * beam_size x seq_len]
            # dec_partial_pos = dec_partial_pos.repeat(n_remaining_sents * beam_size, 1)
            # # wrap into a Variable
            # dec_partial_pos = Variable(dec_partial_pos.type(torch.LongTensor), volatile=True)
            dec_partial_inputs_len = torch.LongTensor(n_remaining_sents,).fill_(len_dec_seq) # TODO: note
            dec_partial_inputs_len = dec_partial_inputs_len.repeat(beam_size)
            #dec_partial_inputs_len = Variable(dec_partial_inputs_len, volatile=True)

            if self.use_cuda:
                dec_partial_inputs = dec_partial_inputs.cuda()
                dec_partial_inputs_len = dec_partial_inputs_len.cuda()

            # Decoding
            dec_outputs, *_ = self.model.decode(dec_partial_inputs, dec_partial_inputs_len,
                                                enc_inputs, enc_outputs) # TODO:
            # print('1',enc_inputs.size(), dec_outputs.size())
            dec_outputs = dec_outputs[:,-1,:] # [batch_size * beam_size x d_model]
            # print('2',dec_outputs.size())
            dec_outputs = self.model.tgt_proj(dec_outputs)
            # print('3',dec_outputs.size())
            out = self.model.prob_proj(dec_outputs)
            # print('4',out.size())

            # [batch_size x beam_size x tgt_vocab_size]
            word_lk = out.view(n_remaining_sents, beam_size, -1).contiguous()

            active_beam_idx_list = []
            for beam_idx in range(batch_size):
                if beams[beam_idx].done:
                    continue

                ########################################################################################
                ########################################################################################
                # initial
                # inst_idx = beam_inst_idx_map[beam_idx] # 해당 beam_idx 의 데이터가 실제 data 에서 몇번째 idx인지
                # if not beams[beam_idx].advance(word_lk.data[inst_idx]):
                #     active_beam_idx_list += [beam_idx]
                ########################################################################################
                ########################################################################################
                inst_idx = beam_inst_idx_map[beam_idx]
                if i < 4:
                    if not beams[beam_idx].advance(word_lk.data[inst_idx]):
                        active_beam_idx_list += [beam_idx]
                else:
                    final_seqs = []
                    source_sent = []
                    for src_sent in enc_inputs:
                        pred_line = convert_idx2text(src_sent, tgt_idx2word)
                        source_sent.append(pred_line)
                        break
                    for idx_seq in dec_partial_inputs:
                        pred_line = convert_idx2text(idx_seq, tgt_idx2word)
                        final_seqs.append(pred_line)
                    if not beams[beam_idx].advance_sub_3(word_lk.data[inst_idx], source_sent, final_seqs, len_dec_seq-1):
                        active_beam_idx_list += [beam_idx]



            if not active_beam_idx_list: # all instances have finished their path to <eos>
                break

            # In this section, the sentences that are still active are
            # compacted so that the decoder is not run on completed sentences
            active_inst_idxs = self.tt.LongTensor(
                [beam_inst_idx_map[k] for k in active_beam_idx_list]) # TODO: fix

            # update the idx mapping
            beam_inst_idx_map = {
                beam_idx: inst_idx for inst_idx, beam_idx in enumerate(active_beam_idx_list)}
            # print("2 beam_inst_idx_map",beam_inst_idx_map)

            def update_active_seq(seq_var, active_inst_idxs):
                ''' Remove the encoder outputs of finished instances in one batch. '''
                inst_idx_dim_size, *rest_dim_sizes = seq_var.size()
                inst_idx_dim_size = inst_idx_dim_size * len(active_inst_idxs) // n_remaining_sents
                new_size = (inst_idx_dim_size, *rest_dim_sizes)

                # select the active instances in batch
                original_seq_data = seq_var.data.view(n_remaining_sents, -1)
                active_seq_data = original_seq_data.index_select(0, active_inst_idxs)
                active_seq_data = active_seq_data.view(*new_size)

                return Variable(active_seq_data, volatile=True)

            def update_active_enc_info(enc_info_var, active_inst_idxs):
                ''' Remove the encoder outputs of finished instances in one batch. '''

                inst_idx_dim_size, *rest_dim_sizes = enc_info_var.size()
                inst_idx_dim_size = inst_idx_dim_size * len(active_inst_idxs) // n_remaining_sents
                new_size = (inst_idx_dim_size, *rest_dim_sizes)

                # select the active instances in batch
                original_enc_info_data = enc_info_var.data.view(
                    n_remaining_sents, -1, self.model_opt.d_model)
                active_enc_info_data = original_enc_info_data.index_select(0, active_inst_idxs)
                active_enc_info_data = active_enc_info_data.view(*new_size)

                return Variable(active_enc_info_data, volatile=True)

            enc_inputs = update_active_seq(enc_inputs, active_inst_idxs)
            enc_outputs = update_active_enc_info(enc_outputs, active_inst_idxs)

            # update the remaining size
            n_remaining_sents = len(active_inst_idxs)

        # Return useful information
        all_hyp, all_scores = [], []
        n_best = self.opt.n_best

        for beam_idx in range(batch_size):
            scores, tail_idxs = beams[beam_idx].sort_scores()
            # print("scores", scores, scores.size())
            # print("tail_idxs", tail_idxs, tail_idxs.size())
            all_scores += [scores[:n_best]]

            hyps = [beams[beam_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
            all_hyp += [hyps]

        return all_hyp, all_scores
