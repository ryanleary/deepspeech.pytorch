import argparse
import json
import torch
from torch.autograd import Variable

from beam_decoder import decode
from data.data_loader import SpectrogramParser
from decoder import ArgMaxDecoder, BeamLMDecoder
from model import DeepSpeech, supported_rnns
import numpy as np
import kenlm

parser = argparse.ArgumentParser(description='DeepSpeech prediction')
parser.add_argument('--sample_rate', default=16000, type=int, help='Sample rate')
parser.add_argument('--labels_path', default='labels.json', help='Contains all characters for prediction')
parser.add_argument('--model_path', default='models/deepspeech_final.pth.tar',
                    help='Path to model file created by training')
parser.add_argument('--audio_path', default='audio.wav',
                    help='Audio file to predict on')
parser.add_argument('--window_size', default=.02, type=float, help='Window size for spectrogram in seconds')
parser.add_argument('--window_stride', default=.01, type=float, help='Window stride for spectrogram in seconds')
parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')
parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')
parser.add_argument('--rnn_type', default='lstm', help='Type of the RNN. rnn|gru|lstm are supported')
parser.add_argument('--bidirectional', dest='bidirectional', action='store_true', help='Use bi-directional rnns')
parser.set_defaults(cuda=False, bidirectional=False)

args = parser.parse_args()


class BeamSearch(object):
    """
    Decoder for audio to text.

    From: https://arxiv.org/pdf/1408.2873.pdf (hardcoded)
    """

    def __init__(self, lm,  alphabet='" abcdefghijklmnopqrstuvwxyz', blank='_'):
        # blank symbol plus alphabet
        self.alphabet = alphabet
        # index of each char
        self.char_to_index = {c: i for i, c in enumerate(self.alphabet)}
        self.blank = blank
        self.lm = lm

    def _get_last_score(self, prefix):
        full_scores = self.lm.full_scores(prefix, eos = False, bos=False)
        probas = [f[0] for f in full_scores ]
        if len(probas) == 0:
            return 1e-15
        return probas[-1]

    def decode(self, probs, k=100):
        """
        Decoder.

        :param probs: matrix of size Windows X AlphaLength
        :param k: beam size
        :returns: most probable prefix in A_prev
        """
        # List of prefixs, initialized with empty char
        A_prev = ['']
        # Probability of a prefix at windows time t to ending in blank
        p_b = {('', 0): 1.0}
        # Probability of a prefix at windows time t to not ending in blank
        p_nb = {('', 0): 0.0}

        # for each time window t
        for t in range(1, probs.shape[0] + 1):
            A_new = []
            # for each prefix
            for s in A_prev:
                for c in self.alphabet:
                    if c == self.blank:
                        p_b[(s, t)] = probs[t - 1][self.char_to_index[self.blank]] * \
                                      (p_b[(s, t - 1)] + p_nb[(s, t - 1)])
                        A_new.append(s)
                    else:
                        s_new = s + c
                        # repeated chars
                        if len(s) > 0 and c == s[-1]:
                            p_nb[(s_new, t)] = probs[t - 1][self.char_to_index[c]] * \
                                               p_b[(s, t - 1)]
                            p_nb[(s, t)] = probs[t - 1][self.char_to_index[c]] * \
                                           p_b[(s, t - 1)]
                        # spaces
                        elif c == ' ':
                            p_nb[(s_new, t)] = self._get_last_score( s_new )*probs[t - 1][self.char_to_index[c]] * \
                                               (p_b[(s, t - 1)] + p_nb[(s, t - 1)])
                        else:
                            p_nb[(s_new, t)] = probs[t - 1][self.char_to_index[c]] * \
                                               (p_b[(s, t - 1)] + p_nb[(s, t - 1)])
                            p_nb[(s, t)] = probs[t - 1][self.char_to_index[c]] * \
                                           (p_b[(s, t - 1)] + p_nb[(s, t - 1)])
                        if s_new not in A_prev:
                            p_b[(s_new, t)] = probs[t - 1][self.char_to_index[self.blank]] * \
                                              (p_b[(s, t - 1)] + p_nb[(s, t - 1)])
                            p_nb[(s_new, t)] = probs[t - 1][self.char_to_index[c]] * \
                                               p_nb[(s, t - 1)]
                        A_new.append(s_new)

            s_probs = map(lambda x: (x, (p_b[(x, t)] + p_nb[(x, t)]) * len(x)), A_new)
            xs = sorted(s_probs, key=lambda x: x[1], reverse=True)[:k]
            A_prev, best_probs = zip(*xs)
        return A_prev[0], best_probs[0]

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

if __name__ == '__main__':
    model = DeepSpeech.load_model(args.model_path, cuda=args.cuda)
    model.eval()

    labels = DeepSpeech.get_labels(model)
    audio_conf = DeepSpeech.get_audio_conf(model)

    decoder = ArgMaxDecoder(labels)

    lm = kenlm.LanguageModel("models/3-gram.pruned.1e-7.bin")
    beam_decoder = BeamLMDecoder(labels, lm)

    parser = SpectrogramParser(audio_conf, normalize=True)
    spect = parser.parse_audio(args.audio_path).contiguous()
    spect = spect.view(1, 1, spect.size(0), spect.size(1))
    out = model(Variable(spect))
    out = out.transpose(0, 1)  # TxNxH

    print("argmax:", [x.lower() for x in decoder.decode(out.data)])
    print("beamlm:", [x.lower() for x in beam_decoder.decode(out.data)])

    #import joblib
    #raw_scores = out.data.cpu().numpy()
    #joblib.dump(raw_scores, "scores.npy")
    #print("arg max:    ", decoded_output[0])

    # scores = np.transpose(raw_scores, axes=(1, 0, 2))
    # scores = scores[0]
    # scores = softmax(scores.T).T
    #
    # lm = kenlm.LanguageModel("models/3-gram.pruned.1e-7.bin")
    # beam = BeamSearch(lm, alphabet=labels, blank = '_',)
    # print(beam.decode( scores, k = 10 ))
    #lm = kenlm.LanguageModel("enwiki9.binary")
    #labels = json.loads(open("labels.json").read())
    #int_char_map = {i: s for i, s in enumerate(labels)}
    #print("4-gram LM model    ", decode(lm, scores, int_char_map, beam=250, alpha=1.0, top_n=1))
