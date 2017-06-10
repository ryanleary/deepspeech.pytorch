#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2015-2016 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
# Modified to support pytorch Tensors

import Levenshtein as Lev
import torch
from six.moves import xrange
import numpy as np


class Decoder(object):
    """
    Basic decoder class from which all other decoders inherit. Implements several
    helper functions. Subclasses should implement the decode() method.

    Arguments:
        labels (string): mapping from integers to characters.
        blank_index (int, optional): index for the blank '_' character. Defaults to 0.
        space_index (int, optional): index for the space ' ' character. Defaults to 28.
    """

    def __init__(self, labels, blank_index=0, space_index=28):
        # e.g. labels = "_'ABCDEFGHIJKLMNOPQRSTUVWXYZ#"
        self.labels = labels
        self.int_to_char = dict([(i, c) for (i, c) in enumerate(labels)])
        self.char_to_int = dict([(c, i) for i, c in self.int_to_char.iteritems()])
        self.blank_index = blank_index
        self.space_index = space_index

    def convert_to_strings(self, sequences, sizes=None):
        """Given a list of numeric sequences, returns the corresponding strings"""
        strings = []
        for x in xrange(len(sequences)):
            string = self.convert_to_string(sequences[x])
            string = string[0:int(sizes.data[x])] if sizes is not None else string
            strings.append(string)
        return strings

    def convert_to_string(self, sequence):
        return ''.join([self.int_to_char[i] for i in sequence])

    def process_strings(self, sequences, remove_repetitions=False):
        """
        Given a list of strings, removes blanks and replace space character with space.
        Option to remove repetitions (e.g. 'abbca' -> 'abca').

        Arguments:
            sequences: list of 1-d array of integers
            remove_repetitions (boolean, optional): If true, repeating characters
                are removed. Defaults to False.
        """
        processed_strings = []
        for sequence in sequences:
            string = self.process_string(remove_repetitions, sequence).strip()
            processed_strings.append(string)
        return processed_strings

    def process_string(self, remove_repetitions, sequence):
        string = ''
        for i, char in enumerate(sequence):
            if char != self.int_to_char[self.blank_index]:
                # if this char is a repetition and remove_repetitions=true,
                # skip.
                if remove_repetitions and i != 0 and char == sequence[i - 1]:
                    pass
                elif char == self.labels[self.space_index]:
                    string += ' '
                else:
                    string = string + char
        return string

    def wer(self, s1, s2):
        """
        Computes the Word Error Rate, defined as the edit distance between the
        two provided sentences after tokenizing to words.
        Arguments:
            s1 (string): space-separated sentence
            s2 (string): space-separated sentence
        """

        # build mapping of words to integers
        b = set(s1.split() + s2.split())
        word2char = dict(zip(b, range(len(b))))

        # map the words to a char array (Levenshtein packages only accepts
        # strings)
        w1 = [chr(word2char[w]) for w in s1.split()]
        w2 = [chr(word2char[w]) for w in s2.split()]

        return Lev.distance(''.join(w1), ''.join(w2))

    def cer(self, s1, s2):
        """
        Computes the Character Error Rate, defined as the edit distance.

        Arguments:
            s1 (string): space-separated sentence
            s2 (string): space-separated sentence
        """
        return Lev.distance(s1, s2)

    def decode(self, probs, sizes=None):
        """
        Given a matrix of character probabilities, returns the decoder's
        best guess of the transcription

        Arguments:
            probs: Tensor of character probabilities, where probs[c,t]
                            is the probability of character c at time t
            sizes(optional): Size of each sequence in the mini-batch
        Returns:
            string: sequence of the model's best guess for the transcription

        """
        raise NotImplementedError

    def strings_to_labels(self, str_list):
        str_list_len = len(str_list)
        # str_list_len_sum = sum([len(str) for str in str_list])
        label_lens = torch.IntTensor(str_list_len)
        labels = None

        for i, str in enumerate(str_list):
            cur_str_len = len(str)
            cur_str = torch.IntTensor(cur_str_len)  # create a tensor to represent labels of string
            for j, c in enumerate(str):
                cur_str[j] = self.char_to_int[c]  # insert indexes of chars into tensor
            # concat the current string tensor with the overall labels
            if labels is None:
                labels = cur_str
            else:
                labels = torch.cat((labels, cur_str), 0)
            label_lens[i] = cur_str_len

        return labels, label_lens


class ArgMaxDecoder(Decoder):
    def decode(self, probs, sizes=None):
        """
        Returns the argmax decoding given the probability matrix. Removes
        repeated elements in the sequence, as well as blanks.

        Arguments:
            probs: Tensor of character probabilities from the network. Expected shape of seq_length x batch x output_dim
            sizes(optional): Size of each sequence in the mini-batch
        Returns:
            strings: sequences of the model's best guess for the transcription on inputs
        """
        _, max_probs = torch.max(probs.transpose(0, 1), 2)
        strings = self.convert_to_strings(max_probs.view(max_probs.size(0), max_probs.size(1)), sizes)
        return self.process_strings(strings, remove_repetitions=True)

class BeamLMDecoder(Decoder):
    def __init__(self, labels, lm, beam_width=20, top_n=1, blank_index=0, space_index=28):
        super(BeamLMDecoder, self).__init__(labels, blank_index=blank_index, space_index=space_index)
        self._lm = lm
        self._beam_width=beam_width
        self._top_n=top_n
        self._alpha = 0.5

    def decode(self, probs, sizes=None):
        probs = probs.transpose(0, 1)
        probs_shape = probs.size()
        S = probs_shape[0]
        N = probs_shape[2]
        T = probs_shape[1]
        s = 0

        candidates = [ (0.0, '', 0.0) ]
        #(acoustic_proba, prefix, lm)
        for t in range(T):
            new_candidates = []
            for i in range(N):
                symbol = self.int_to_char[i]
                # if symbol == '_':
                #     symbol = ''
                #acoustic_proba = np.log1p()
                for candidate in candidates:
                    prefix_acoustic_score, prefix, _ = candidate
                    new_phrase = prefix + symbol
                    new_phrase = new_phrase.upper()
                    #print(new_phrase, s,i,t)
                    lm_score = self._lm.score(new_phrase, bos=False, eos=False)
                    #print(self.process_string(True, prefix) + symbol, lm_score)

                    total_acoustic_score =  prefix_acoustic_score + probs[s][t][i]
                    total_score = (1-self._alpha)*total_acoustic_score + self._alpha*lm_score

                    new_candidates.append( ( total_acoustic_score, new_phrase, total_score ) )
            sorted_new_candidates = sorted( new_candidates, key = lambda x: x[-1], reverse=True )
            candidates = sorted_new_candidates[:self._beam_width]
        return self.process_strings([c[1] for c in candidates[:self._top_n]], remove_repetitions=True)

class BeamSearchDecoder(Decoder):
    def __init__(self, labels, beam_size=12):
        super(BeamSearchDecoder, self).__init__(labels=labels)
        self.beam_size = beam_size

    def decode(self, probs, sizes=None):
        from ctc_beamsearch import ctc_beamsearch
        strings = []
        probs_transpose = probs.transpose(0, 1).cpu()

        # iterate over probability distribution in each batch [due to API of ctc_beamsearch]
        for batch_idx in xrange(probs.size(1)):
            b_probs = probs_transpose[batch_idx].numpy()
            decoded = ctc_beamsearch(b_probs, alphabet=self.labels, blank_symbol='_', k=self.beam_size)
            strings.append(decoded)
        return self.process_strings(strings, remove_repetitions=True)
