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
import os
import kenlm
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
        return [[(1.0, x)] for x in self.process_strings(strings, remove_repetitions=True)]


class Scorer(object):
    def evaluate(self, sentence):
        raise NotImplementedError


class KenLMScorer(Scorer):
    """
    External defined scorer to evaluate a sentence in beam search
               decoding, consisting of language model and word count.

    :param alpha: Parameter associated with language model.
    :type alpha: float
    :param beta: Parameter associated with word count.
    :type beta: float
    :model_path: Path to load language model.
    :type model_path: basestring
    """

    def __init__(self, alpha, beta, model_path, vocab_path=None):
        self._alpha = alpha
        self._beta = beta
        if not os.path.isfile(model_path):
            raise IOError("Invalid language model path: %s" % model_path)
        self._language_model = kenlm.LanguageModel(model_path)
        self._vocab_scorer = VocabularyScorer(vocab_path) if vocab_path is not None else None


    # n-gram language model scoring
    def language_model_score(self, sentence):
        #log prob of last word
        log_cond_prob = list(
            self._language_model.full_scores(sentence, eos=False))[-1][0]
        return np.power(10, log_cond_prob)

    # word insertion term
    def word_count(self, sentence):
        words = sentence.strip().split()
        return len(words)

    # execute evaluation
    def evaluate(self, sentence):
        lm = self.language_model_score(sentence)
        word_cnt = self.word_count(sentence)
        vocab_score = self._vocab_scorer.evaluate(sentence) if self._vocab_scorer is not None else 1
        score = vocab_score * np.power(lm, self._alpha) \
                * np.power(word_cnt, self._beta)
        return score

class VocabularyScorer(object):
    def __init__(self, vocab_path):
        if not os.path.isfile(vocab_path):
            raise IOError("Invalid dictionary path: %s" % vocab_path)
        self._vocab = set([])
        with open(vocab_path, 'r') as fh:
            for line in fh:
                self._vocab.add(line.strip())

    # execute evaluation
    def evaluate(self, sentence):
        # lm = self.language_model_score(sentence)
        # word_cnt = self.word_count(sentence)
        # score = np.power(lm, self._alpha) \
        #         * np.power(word_cnt, self._beta)
        words = sentence.strip().split()
        if len(words) > 0:
            if words[-1] in self._vocab:
                return 1
            else:
                return 0.0000001
        else:
            return 1
        return score


class PrefixBeamCTCDecoder(Decoder):
    def __init__(self, labels, scorer, beam_width=20, top_n=1, blank_index=0, space_index=28):
        super(PrefixBeamCTCDecoder, self).__init__(labels, blank_index=blank_index, space_index=space_index)
        self._beam_width = beam_width
        self.char_to_int = dict([(c, i) for (i, c) in enumerate(labels)])
        self._top_n = top_n
        self._alpha = 0.5
        self._scorer = scorer

    def decode(self, probs, sizes=None):
        probs = probs.transpose(0, 1)
        probs_shape = probs.size()
        # print(probs_shape)
        S = probs_shape[0]  # number of utterances to decode
        T = probs_shape[1]  # number of timesteps
        N = probs_shape[2]  # number of output classes (labels)

        results = []
        for s in range(S):
            # initialize
            # the set containing selected prefixes
            prefix_set_prev = {'\t': 1.0}
            probs_b_prev, probs_nb_prev = {'\t': 1.0}, {'\t': 0.0}
            probs_s = torch.nn.functional.softmax(torch.autograd.Variable(probs[s], volatile=True)).data
            # extend prefix in loop
            seq_len = T if sizes is None else sizes.data[s]
            for time_step in range(seq_len):
                # the set containing candidate prefixes
                prefix_set_next = {}
                probs_b_cur, probs_nb_cur = {}, {}
                for l in prefix_set_prev:
                    prob = probs_s[time_step]
                    if l not in prefix_set_next:
                        probs_b_cur[l], probs_nb_cur[l] = 0.0, 0.0

                    # extend prefix by travering vocabulary
                    for c in range(0, N):
                        if c == self.blank_index:
                            probs_b_cur[l] += prob[c] * (
                                probs_b_prev[l] + probs_nb_prev[l])
                        else:
                            last_char = l[-1]
                            new_char = self.int_to_char[c]
                            l_plus = l + new_char
                            if l_plus not in prefix_set_next:
                                probs_b_cur[l_plus], probs_nb_cur[l_plus] = 0.0, 0.0

                            if new_char == last_char:
                                probs_nb_cur[l_plus] += prob[c] * probs_b_prev[l]
                                probs_nb_cur[l] += prob[c] * probs_nb_prev[l]
                            elif new_char == ' ':
                                if (self._scorer is None) or (len(l) == 1):
                                    score = 1.0
                                else:
                                    prefix = l[1:]
                                    score = self._scorer.evaluate(prefix)
                                probs_nb_cur[l_plus] += score * prob[c] * (
                                    probs_b_prev[l] + probs_nb_prev[l])
                            else:
                                probs_nb_cur[l_plus] += prob[c] * (
                                    probs_b_prev[l] + probs_nb_prev[l])
                            # add l_plus into prefix_set_next
                            prefix_set_next[l_plus] = probs_nb_cur[
                                                          l_plus] + probs_b_cur[l_plus]
                    # add l into prefix_set_next
                    prefix_set_next[l] = probs_b_cur[l] + probs_nb_cur[l]
                # update probs
                probs_b_prev, probs_nb_prev = probs_b_cur, probs_nb_cur

                # store top beam_size prefixes
                prefix_set_prev = sorted(
                    prefix_set_next.items(), key=lambda asd: asd[1], reverse=True)
                if self._beam_width < len(prefix_set_prev):
                    prefix_set_prev = prefix_set_prev[:self._beam_width]
                prefix_set_prev = dict(prefix_set_prev)

            beam_result = []
            for (seq, prob) in prefix_set_prev.items():
                if prob > 0.0:
                    result = seq[1:]
                    log_prob = np.log(prob)
                    beam_result.append((log_prob, result))

            # output top beam_size decoding results
            beam_result = sorted(beam_result, key=lambda x: x[0], reverse=True)[0:min(self._beam_width, self._top_n)]
            results.append(beam_result)
        return results
