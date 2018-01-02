import argparse
import errno
import json
import os
import time

import torch
from tqdm import tqdm
from torch.autograd import Variable
from warpctc_pytorch import CTCLoss
from data.data_loader import AudioDataLoader, SpectrogramDataset, BucketingSampler
from decoder import GreedyDecoder
from model import DeepSpeech, supported_rnns
from logger import BaseLogger, TensorboardLogger, VizLogger

parser = argparse.ArgumentParser(description='DeepSpeech training')
parser.add_argument('--train-manifest', metavar='DIR',
                    help='path to train manifest', default='data/train_manifest.jl')
parser.add_argument('--val-manifest', metavar='DIR',
                    help='path to validation manifest', default='data/val_manifest.jl')
parser.add_argument('--sample-rate', default=16000, type=int, help='Sample rate')
parser.add_argument('--batch-size', default=20, type=int, help='Batch size for training')
parser.add_argument('--num-workers', default=4, type=int, help='Number of workers used in data-loading')
parser.add_argument('--labels-path', default='labels.json', help='Contains all characters for transcription')
parser.add_argument('--window-size', default=.02, type=float, help='Window size for spectrogram in seconds')
parser.add_argument('--window-stride', default=.01, type=float, help='Window stride for spectrogram in seconds')
parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')
parser.add_argument('--hidden-size', default=800, type=int, help='Hidden size of RNNs')
parser.add_argument('--hidden-layers', default=5, type=int, help='Number of RNN layers')
parser.add_argument('--rnn-type', default='gru', help='Type of the RNN. rnn|gru|lstm are supported')
parser.add_argument('--epochs', default=70, type=int, help='Number of training epochs')
parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--max-norm', default=400, type=int, help='Norm cutoff to prevent explosion of gradients')
parser.add_argument('--learning-anneal', default=1.1, type=float, help='Annealing applied to learning rate every epoch')
parser.add_argument('--silent', dest='silent', action='store_true', help='Turn off progress tracking per iteration')
parser.add_argument('--checkpoint', dest='checkpoint', action='store_true', help='Enables checkpoint saving of model')
parser.add_argument('--checkpoint-per-batch', default=0, type=int, help='Save checkpoint per batch. 0 means never save')
parser.add_argument('--visdom', dest='visdom', action='store_true', help='Turn on visdom graphing')
parser.add_argument('--tensorboard', dest='tensorboard', action='store_true', help='Turn on tensorboard graphing')
parser.add_argument('--log-dir', default='visualize/deepspeech_final', help='Location of tensorboard log')
parser.add_argument('--log-params', dest='log_params', action='store_true', help='Log parameter values and gradients')
parser.add_argument('--id', default='Deepspeech training', help='Identifier for visdom/tensorboard run')
parser.add_argument('--save-folder', default='models/', help='Location to save epoch models')
parser.add_argument('--model-path', default='models/deepspeech_final.pth.tar',
                    help='Location to save best validation model')
parser.add_argument('--continue-from', default='', help='Continue from checkpoint model')
parser.add_argument('--finetune', dest='finetune', action='store_true',
                    help='Finetune the model from checkpoint "continue_from"')
parser.add_argument('--augment-config', type=str, default=None, help='Path to data augmentation configuration file')
parser.add_argument('--no-shuffle', dest='no_shuffle', action='store_true',
                    help='Turn off shuffling and sample from dataset based on sequence length (smallest to largest)')
parser.add_argument('--no-bidirectional', dest='bidirectional', action='store_false', default=True,
                    help='Turn off bi-directional RNNs, introduces lookahead convolution')
parser.add_argument('--max-elements', type=int, default=0,
                    help='maximum number of elements in a training input matrix, larger matrices will result in subbatching')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_subbatches(data_tuple, nominal_batch_size, max_size=0):
    if max_size == 0:
        yield data_tuple
    else:
        (a, b, c, d) = data_tuple
        shape = a.size()
        max_batch_size = min(nominal_batch_size, int(max_size//(shape[1]*shape[2]*shape[3]))-1)
        if max_batch_size < nominal_batch_size:
            print("  Warn: Batch too large. Splitting into subbatches with maxsize =", max_batch_size)
            for i in range(0, shape[0], max_batch_size):
                yield (a[i:i+max_batch_size],
                       b[i:i+max_batch_size],
                       c[i:i+max_batch_size],
                       d[i:i+max_batch_size])
        else:
            yield data_tuple

def create_dir_safe(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            print('Model Save directory already exists.')
        else:
            raise


def load_model(path, epochs, finetune=False, viz=BaseLogger()):
    package = torch.load(path, map_location=lambda storage, loc: storage)
    model = DeepSpeech.load_model_package(package)
    labels = DeepSpeech.get_labels(model)
    audio_conf = DeepSpeech.get_audio_conf(model)
    parameters = model.parameters()
    optimizer = torch.optim.SGD(parameters, lr=args.lr,
                                momentum=args.momentum, nesterov=True)
    ts = TrainStats(epochs)
    if not finetune:  # Don't want to restart training
        optimizer.load_state_dict(package['optim_dict'])
        ts.start_epoch = int(package.get('epoch', 1)) - 1  # Index start at 0 for training
        ts.start_iter = package.get('iteration', None)
        if ts.start_iter is None:
            ts.start_epoch += 1  # We saved model after epoch finished, start at the next epoch.
            ts.start_iter = 0
        else:
            ts.start_iter += 1
        ts.avg_loss = int(package.get('avg_loss', 0))
        ts.loss_results, ts.cer_results, ts.wer_results = package['loss_results'], package[
            'cer_results'], package['wer_results']
        if package['loss_results'] is not None and start_epoch > 0:
            viz.log_previous_epochs(start_epoch, loss_results, wer_results, cer_results)
    return model, labels, audio_conf, optimizer, ts

def init_model(args):
    with open(args.labels_path) as label_file:
        labels = str(''.join(json.load(label_file)))

    ts = TrainStats(args.epochs)
    audio_conf = dict(sample_rate=args.sample_rate,
                      window_size=args.window_size,
                      window_stride=args.window_stride,
                      window=args.window)

    rnn_type = args.rnn_type.lower()
    assert rnn_type in supported_rnns, "rnn_type should be either lstm, rnn or gru"
    model = DeepSpeech(rnn_hidden_size=args.hidden_size,
                       nb_layers=args.hidden_layers,
                       labels=labels,
                       rnn_type=supported_rnns[rnn_type],
                       audio_conf=audio_conf,
                       bidirectional=args.bidirectional)
    parameters = model.parameters()
    optimizer = torch.optim.SGD(parameters, lr=args.lr,
                                momentum=args.momentum, nesterov=True)
    return model, labels, audio_conf, optimizer, ts

class TrainStats(object):
    def __init__(self, epochs):
        self.loss_results = torch.Tensor(epochs)
        self.cer_results = torch.Tensor(epochs)
        self.wer_results = torch.Tensor(args.epochs)
        self.best_wer = None
        self.avg_loss = 0
        self.start_epoch = 0
        self.start_iter = 0

def evaluate(test_loader, model):
    model.eval()
    total_cer, total_wer, total_loss = 0, 0, 0
    for i, (data) in tqdm(enumerate(test_loader), total=len(test_loader)):
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

        target_sizes = Variable(target_sizes, requires_grad=False)
        targets = Variable(targets, requires_grad=False)

        out = model(inputs)
        out = out.transpose(0, 1)  # TxNxH
        seq_length = out.size(0)
        sizes = Variable(input_percentages.mul_(int(seq_length)).int(), requires_grad=False)

        total_loss += criterion(out, targets, sizes, target_sizes)

        decoded_output, _ = decoder.decode(out.data, sizes.data)
        target_strings = decoder.convert_to_strings(split_targets)
        wer, cer = 0, 0
        for x in range(len(target_strings)):
            transcript, reference = decoded_output[x][0], target_strings[x][0]
            wer += decoder.wer(transcript, reference) / float(len(reference.split()))
            cer += decoder.cer(transcript, reference) / float(len(reference))
        total_cer += cer
        total_wer += wer

        if args.cuda:
            torch.cuda.synchronize()
        del out
        del inputs
        del targets
        del input_percentages
        del target_sizes
        del split_targets
    wer = (total_wer / len(test_loader.dataset)) * 100
    cer = (total_cer / len(test_loader.dataset)) * 100
    loss = (total_loss / len(test_loader.dataset))
    return wer, cer, loss

if __name__ == '__main__':
    torch.manual_seed(123456)
    torch.cuda.manual_seed_all(123456)
    args = parser.parse_args()

    create_dir_safe(args.save_folder)

    if args.continue_from:  # Starting from previous model
        print("Loading checkpoint model %s" % args.continue_from)
        model, labels, audio_conf, optimizer, ts = load_model(args.continue_from, args.epochs, finetune=args.finetune, viz=viz)
    else:
        model, labels, audio_conf, optimizer, ts = init_model(args)

    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()

    viz = BaseLogger()
    if args.visdom:
        viz = VizLogger(args.id, args.epochs)
    if args.tensorboard:
        viz = TensorboardLogger(args.id, args.log_dir, model=model)

    print(model)
    print("Number of parameters: %d" % DeepSpeech.get_param_size(model))

    # set up loss function and decoder
    criterion = CTCLoss()
    decoder = GreedyDecoder(labels)

    # set up data loaders
    train_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.train_manifest, labels=labels,
                                       normalize=True, augment_config=args.augment_config, min_duration=1.0, max_duration=18.0)
    test_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.val_manifest, labels=labels,
                                      normalize=True, augment_config=None)
    train_sampler = BucketingSampler(train_dataset, batch_size=args.batch_size)
    train_loader = AudioDataLoader(train_dataset,
                                   num_workers=args.num_workers, batch_sampler=train_sampler)
    test_loader = AudioDataLoader(test_dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers)

    if not args.no_shuffle and ts.start_epoch != 0:
        print("Shuffling batches for the following epochs")
        train_sampler.shuffle()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    wer, cer, loss = evaluate(test_loader, model)
    viz.init_epoch(loss, wer, cer, loss)

    # iterate over dataset an epoch at a time
    step = 0
    for epoch in range(ts.start_epoch, args.epochs):
        model.train()
        end = time.time()
        for i, (data) in enumerate(train_loader, start=ts.start_iter):
            if i == len(train_sampler):
                break
            # measure data loading time
            data_time.update(time.time() - end)
            for inputs, targets, input_percentages, target_sizes in get_subbatches(data, args.batch_size, max_size=args.max_elements):
                subbatch_size = inputs.size(0)
                inputs = Variable(inputs, requires_grad=False)
                target_sizes = Variable(target_sizes, requires_grad=False)
                targets = Variable(targets, requires_grad=False)

                if args.cuda:
                    inputs = inputs.cuda()
                if args.cuda and epoch == 0 and i % 2 == 0:
                    torch.cuda.empty_cache()

                out = model(inputs)
                out = out.transpose(0, 1)  # TxNxH

                seq_length = out.size(0)
                sizes = Variable(input_percentages.mul_(int(seq_length)).int(), requires_grad=False)

                loss = (criterion(out, targets, sizes, target_sizes) / args.batch_size)
                del out
                del inputs
                del target_sizes
                del targets
                del seq_length
                del sizes

                loss_sum = loss.data.sum()
                inf = float("inf")
                if loss_sum == inf or loss_sum == -inf:
                    print("WARNING: received an inf loss, setting loss value to 0")
                    loss_value = 0
                else:
                    loss_value = loss.data[0]

                ts.avg_loss += loss_value
                losses.update(loss_value, subbatch_size)

                # compute gradient
                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm(model.parameters(), args.max_norm)
                # SGD step
                optimizer.step()
                del loss

                if args.cuda:
                    torch.cuda.synchronize()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            viz.log_step(step, losses.val)
            step += 1
            if not args.silent:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    (epoch + 1), (i + 1), len(train_sampler), batch_time=batch_time,
                    data_time=data_time, loss=losses))
            if args.checkpoint_per_batch > 0 and i > 0 and (i + 1) % args.checkpoint_per_batch == 0:
                file_path = '%s/deepspeech_checkpoint_epoch_%d_iter_%d.pth.tar' % (args.save_folder, epoch + 1, i + 1)
                print("Saving checkpoint model to %s" % file_path)
                torch.save(DeepSpeech.serialize(model, optimizer=optimizer, epoch=epoch, iteration=i,
                                                loss_results=ts.loss_results,
                                                wer_results=ts.wer_results, cer_results=ts.cer_results, avg_loss=ts.avg_loss),
                           file_path)
        ts.avg_loss /= len(train_sampler)

        print('Training Summary Epoch: [{0}]\tAverage Loss {loss:.3f}\t'.format(
            epoch + 1, loss=ts.avg_loss))

        start_iter = 0  # Reset start iteration for next epoch

        wer, cer, loss = evaluate(test_loader, model)

        ts.loss_results[epoch] = ts.avg_loss
        ts.wer_results[epoch] = wer
        ts.cer_results[epoch] = cer
        print('Validation Summary Epoch: [{0}]\t'
              'Average WER {wer:.3f}\t'
              'Average CER {cer:.3f}\t'.format(
            epoch + 1, wer=wer, cer=cer))

        viz.log_epoch(epoch, ts.loss_results, ts.wer_results, ts.cer_results, eval_loss=loss)

        if args.checkpoint:
            file_path = '%s/deepspeech_%d.pth.tar' % (args.save_folder, epoch + 1)
            torch.save(DeepSpeech.serialize(model, optimizer=optimizer, epoch=epoch, loss_results=ts.loss_results,
                                            wer_results=ts.wer_results, cer_results=ts.cer_results),
                       file_path)
        # anneal lr
        optim_state = optimizer.state_dict()
        optim_state['param_groups'][0]['lr'] = optim_state['param_groups'][0]['lr'] / args.learning_anneal
        optimizer.load_state_dict(optim_state)
        print('Learning rate annealed to: {lr:.6f}'.format(lr=optim_state['param_groups'][0]['lr']))

        if ts.best_wer is None or ts.best_wer > wer:
            print("Found better validated model, saving to %s" % args.model_path)
            torch.save(DeepSpeech.serialize(model, optimizer=optimizer, epoch=epoch, loss_results=ts.loss_results,
                                            wer_results=ts.wer_results, cer_results=ts.cer_results)
                       , args.model_path)
            ts.best_wer = wer

        avg_loss = 0
        if not args.no_shuffle:
            print("Shuffling batches...")
            train_sampler.shuffle()
