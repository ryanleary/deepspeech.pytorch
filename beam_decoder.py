from decoder import Decoder, ArgMaxDecoder
import kenlm
import collections
import numpy as np
import math
import json



def decode( lm, scores, int_char_map, beam = 40, alpha = 1.0, top_n = 10):
    N = scores.shape[0]
    T = scores.shape[1]

    candidates = [ (0.0, '', 0.0) ]
    #(acoustic_proba, prefix, lm)
    for t in range(T):
        new_candidates = []
        for i in range(N):
            symbol = int_char_map[i]
            if symbol == '_':
                symbol = ''
            acoustic_proba = np.log(scores[i][t])
            for candidate in candidates:
                prefix_acoustic_score, prefix, _ = candidate
                new_phrase = prefix + symbol
                new_phrase = new_phrase.lower()
                lm_score = lm.score( new_phrase, bos = False, eos = False )

                total_acoustic_score =  prefix_acoustic_score + acoustic_proba
                total_score = total_acoustic_score + alpha*lm_score

                new_candidates.append( ( total_acoustic_score, new_phrase, total_score ) )
        sorted_new_candidates = sorted( new_candidates, key = lambda x: x[-1], reverse=True )
        candidates = sorted_new_candidates[:beam]
    return [c[1] for c in candidates[:top_n]]


if __name__ == "__main__":
    import joblib

    labels = json.loads(open("labels.json").read())
    int_char_map = {i: s for i, s in enumerate(labels)}
    char_int_map = {s: i for i, s in int_char_map.items()}

    raw_scores = joblib.load("scores.npy")
    scores = np.transpose(raw_scores, axes=(1, 0, 2))
    scores = scores[0]
    scores = softmax(scores.T)
    with open("labels.json") as label_file:
        str_labels = str(''.join(json.load(label_file)))

    argmax = ArgMaxDecoder(str_labels)
    print(argmax.decode( raw_scores ))
    #scores = scores.copy(order='F').astype(np.double)

    #dec_argmax = ArgmaxDecoder()
    #dec_argmax.load_chars(int_char_map=int_char_map, char_int_map=char_int_map)
    #hyp_argmax, score_argmax = dec_argmax.decode(scores.T)
    #print(hyp_argmax, score_argmax)

    lm = kenlm.LanguageModel("enwiki9.binary")
    print( decode( lm, scores, int_char_map,  beam=1000, alpha=1.0) )
    #lm_decoder = BeamLMDecoder()
    #lm_decoder.load_chars(int_char_map=int_char_map, char_int_map=char_int_map)
    #lm_decoder.load_lm("enwiki9.binary")
    #hyp_lm_beam, score_lm_beam = lm_decoder.decode(scores.T, beam = 300)
    #print(hyp_lm_beam, score_lm_beam)
    #lm = kenlm.LanguageModel("enwiki9.binary")
    #print(lm.order)
    #scores = np.transpose(joblib.load("scores.npy"), axes = (1,0,2))
    #labels = json.loads(open("labels.json").read())
    #int_char_map = {i : s for i, s in enumerate(labels) }
    #print(scores.shape)
    #print(decode( scores[0],int_char_map,  beam=2000, alpha=0.5))
