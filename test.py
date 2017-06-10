import argparse
import json

import torch
from torch.autograd import Variable

from data.data_loader import SpectrogramDataset, AudioDataLoader
from decoder import ArgMaxDecoder
from model import DeepSpeech

parser = argparse.ArgumentParser(description='DeepSpeech prediction')
parser.add_argument('--sample_rate', default=16000, type=int, help='Sample rate')
parser.add_argument('--labels_path', default='labels.json', help='Contains all characters for prediction')
parser.add_argument('--model_path', default='models/deepspeech_final.pth.tar',
                    help='Path to model file created by training')
parser.add_argument('--window_size', default=.02, type=float, help='Window size for spectrogram in seconds')
parser.add_argument('--window_stride', default=.01, type=float, help='Window stride for spectrogram in seconds')
parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')
parser.add_argument('--cuda', action="store_true", help='Use cuda to test model')
parser.add_argument('--val_manifest', metavar='DIR',
                    help='path to validation manifest csv', default='data/val_manifest.csv')
parser.add_argument('--batch_size', default=20, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--beam_size', dest='beam_size', default=12, type=int, help='Size of beam in decoder beam search.')
parser.add_argument('--decoder', default='argmax', help='Type of the decoder used for inference.')
args = parser.parse_args()

if __name__ == '__main__':
    model = DeepSpeech.load_model(args.model_path, cuda=args.cuda)
    model.eval()

    labels = DeepSpeech.get_labels(model)
    audio_conf = DeepSpeech.get_audio_conf(model)

    if args.decoder == 'beamsearch':
        decoder = BeamSearchDecoder(labels, beam_size=args.beam_size)
    else:
        decoder = ArgMaxDecoder(labels)

    test_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.val_manifest, labels=labels,
                                      normalize=True)
    test_loader = AudioDataLoader(test_dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers)
    total_cer, total_wer = 0, 0
    for i, (data) in enumerate(test_loader):
        inputs, targets, input_percentages, target_sizes = data

        inputs = Variable(inputs)

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
        sizes = Variable(input_percentages.mul_(int(seq_length)).int())

        decoded_output = decoder.decode(out.data, sizes)
        target_strings = decoder.process_strings(decoder.convert_to_strings(split_targets))
        wer, cer = 0, 0
        for x in range(len(target_strings)):
            wer += decoder.wer(decoded_output[x], target_strings[x]) / float(len(target_strings[x].split()))
            cer += decoder.cer(decoded_output[x], target_strings[x]) / float(len(target_strings[x]))
        total_cer += cer
        total_wer += wer

    wer = total_wer / len(test_loader.dataset)
    cer = total_cer / len(test_loader.dataset)

    print('Validation Summary \t'
          'Average WER {wer:.3f}\t'
          'Average CER {cer:.3f}\t'.format(wer=wer * 100, cer=cer * 100))
