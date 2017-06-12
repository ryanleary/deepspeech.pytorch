import argparse
import json

import torch
from torch.autograd import Variable

from data.data_loader import SpectrogramDataset, AudioDataLoader
from decoder import ArgMaxDecoder, PrefixBeamCTCDecoder, KenLMScorer, VocabularyScorer
from model import DeepSpeech

parser = argparse.ArgumentParser(description='DeepSpeech prediction')
parser.add_argument('--model_path', default='models/deepspeech_final.pth.tar',
                    help='Path to model file created by training')
parser.add_argument('--cuda', action="store_true", help='Use cuda to test model')
parser.add_argument('--val_manifest', metavar='DIR',
                    help='path to validation manifest csv', default='data/val_manifest.csv')
parser.add_argument('--batch_size', default=20, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--decoder', default="argmax", choices=["argmax", "beam"], type=str, help="Decoder to use")
beam_args = parser.add_argument_group("Beam Decode Options", "Configurations options for the CTC Beam Search decoder")
beam_args.add_argument('--beam_width', default=10, type=int, help='Beam width to use')
beam_args.add_argument('--lm_path', default=None, type=str, help='Path to an (optional) kenlm language model for use with beam search')
beam_args.add_argument('--lm_alpha', default=0.8, type=float, help='Language model weight')
beam_args.add_argument('--lm_beta', default=4, type=float, help='Language model word penalty')
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

    test_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.val_manifest, labels=labels,
                                      normalize=True)
    test_loader = AudioDataLoader(test_dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers)
    total_cer, total_wer = 0, 0
    for i, (data) in enumerate(test_loader):
        inputs, targets, input_percentages, target_sizes = data

        inputs = Variable(inputs, volatile=True)

        # unflatten targets
        split_targets = []
        offset = 0
        for size in target_sizes:
            split_targets.append(targets[offset:offset + size])
            offset += size

        if args.cuda:
            inputs = inputs.cuda()

        out = model(inputs)
        out = out.transpose(0, 1)  # TxNxH
        seq_length = out.size(0)
        sizes = Variable(input_percentages.mul_(int(seq_length)).int(), volatile=True)

        decoded_output = decoder.decode(out.data, sizes)
        target_strings = decoder.process_strings(decoder.convert_to_strings(split_targets))
        wer, cer = 0, 0
        for x in range(len(target_strings)):
            wer += decoder.wer(decoded_output[x][0][1], target_strings[x]) / float(len(target_strings[x].split()))
            cer += decoder.cer(decoded_output[x][0][1], target_strings[x]) / float(len(target_strings[x]))
        total_cer += cer
        total_wer += wer

    wer = total_wer / len(test_loader.dataset)
    cer = total_cer / len(test_loader.dataset)

    print('Validation Summary \t'
          'Average WER {wer:.3f}\t'
          'Average CER {cer:.3f}\t'.format(wer=wer * 100, cer=cer * 100))
