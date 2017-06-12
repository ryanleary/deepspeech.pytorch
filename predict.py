import argparse

import torch
from torch.autograd import Variable

from data.data_loader import SpectrogramParser
from decoder import ArgMaxDecoder, PrefixBeamCTCDecoder, KenLMScorer, VocabularyScorer
from model import DeepSpeech

parser = argparse.ArgumentParser(description='DeepSpeech prediction')
parser.add_argument('--model_path', default='models/deepspeech_final.pth.tar',
                    help='Path to model file created by training')
parser.add_argument('--audio_path', default='audio.wav',
                    help='Audio file to predict on')
parser.add_argument('--cuda', action="store_true", help='Use cuda to test model')
parser.add_argument('--decoder', default="argmax", choices=["argmax", "beam"], type=str, help="Decoder to use")
beam_args = parser.add_argument_group("Beam Decode Options", "Configurations options for the CTC Beam Search decoder")
beam_args.add_argument('--beam_width', default=10, type=int, help='Beam width to use')
beam_args.add_argument('--lm_path', default=None, type=str, help='Path to an (optional) KenLM language model for use with beam search')
beam_args.add_argument('--lm_alpha', default=1.25, type=float, help='Language model weight (for KenLM)')
beam_args.add_argument('--lm_beta', default=1.5, type=float, help='Language model word penalty (for KenLM)')
beam_args.add_argument('--vocab_path', default=None, type=str, help='Path to an (optional) dictionary file to constrain lexicon')
args = parser.parse_args()

if __name__ == '__main__':
    model = DeepSpeech.load_model(args.model_path, cuda=args.cuda)
    model.eval()

    labels = DeepSpeech.get_labels(model)
    audio_conf = DeepSpeech.get_audio_conf(model)

    if args.decoder == "beam":
        scorer = None
        if args.lm_path is not None:
            scorer = KenLMScorer(args.lm_alpha, args.lm_beta, args.lm_path, args.vocab_path)
        if args.vocab_path is not None:
            scorer = VocabularyScorer(args.vocab_path)
        decoder = PrefixBeamCTCDecoder(labels, scorer, beam_width=args.beam_width, top_n=1, blank_index=labels.index('_'), space_index=labels.index(' '))
    else:
        decoder = ArgMaxDecoder(labels)

    parser = SpectrogramParser(audio_conf, normalize=True)
    spect = parser.parse_audio(args.audio_path).contiguous()
    spect = spect.view(1, 1, spect.size(0), spect.size(1))
    out = model(Variable(spect, volatile=True))
    out = out.transpose(0, 1)  # TxNxH
    decoded_output = decoder.decode(out.data)

    print(decoded_output[0][0][1])
