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

parser = argparse.ArgumentParser(description='DeepSpeech training')
parser.add_argument('--train_manifest', metavar='DIR',
                    help='path to train manifest csv', default='data/train_manifest.csv')
parser.add_argument('--val_manifest', metavar='DIR',
                    help='path to validation manifest csv', default='data/val_manifest.csv')
parser.add_argument('--sample_rate', default=16000, type=int, help='Sample rate')
parser.add_argument('--batch_size', default=20, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in data-loading')
parser.add_argument('--labels_path', default='labels.json', help='Contains all characters for transcription')
parser.add_argument('--window_size', default=.02, type=float, help='Window size for spectrogram in seconds')
parser.add_argument('--window_stride', default=.01, type=float, help='Window stride for spectrogram in seconds')
parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')
parser.add_argument('--hidden_size', default=800, type=int, help='Hidden size of RNNs')
parser.add_argument('--hidden_layers', default=5, type=int, help='Number of RNN layers')
parser.add_argument('--rnn_type', default='gru', help='Type of the RNN. rnn|gru|lstm are supported')
parser.add_argument('--epochs', default=70, type=int, help='Number of training epochs')
parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--max_norm', default=400, type=int, help='Norm cutoff to prevent explosion of gradients')
parser.add_argument('--learning_anneal', default=1.1, type=float, help='Annealing applied to learning rate every epoch')
parser.add_argument('--silent', dest='silent', action='store_true', help='Turn off progress tracking per iteration')
parser.add_argument('--checkpoint', dest='checkpoint', action='store_true', help='Enables checkpoint saving of model')
parser.add_argument('--checkpoint_per_batch', default=0, type=int, help='Save checkpoint per batch. 0 means never save')
parser.add_argument('--visdom', dest='visdom', action='store_true', help='Turn on visdom graphing')
parser.add_argument('--tensorboard', dest='tensorboard', action='store_true', help='Turn on tensorboard graphing')
parser.add_argument('--log_dir', default='visualize/deepspeech_final', help='Location of tensorboard log')
parser.add_argument('--log_params', dest='log_params', action='store_true', help='Log parameter values and gradients')
parser.add_argument('--id', default='Deepspeech training', help='Identifier for visdom/tensorboard run')
parser.add_argument('--save_folder', default='models/', help='Location to save epoch models')
parser.add_argument('--model_path', default='models/deepspeech_final.pth.tar',
                    help='Location to save best validation model')
parser.add_argument('--continue_from', default='', help='Continue from checkpoint model')
parser.add_argument('--finetune', dest='finetune', action='store_true',
                    help='Finetune the model from checkpoint "continue_from"')
parser.add_argument('--augment', dest='augment', action='store_true', help='Use random tempo and gain perturbations.')
parser.add_argument('--noise_dir', default=None,
                    help='Directory to inject noise into audio. If default, noise Inject not added')
parser.add_argument('--noise_prob', default=0.4, help='Probability of noise being added per sample')
parser.add_argument('--noise_min', default=0.0,
                    help='Minimum noise level to sample from. (1.0 means all noise, not original signal)', type=float)
parser.add_argument('--noise_max', default=0.5,
                    help='Maximum noise levels to sample from. Maximum 1.0', type=float)
parser.add_argument('--no_shuffle', dest='no_shuffle', action='store_true',
                    help='Turn off shuffling and sample from dataset based on sequence length (smallest to largest)')
parser.add_argument('--no_bidirectional', dest='bidirectional', action='store_false', default=True,
                    help='Turn off bi-directional RNNs, introduces lookahead convolution')


def to_np(x):
    return x.data.cpu().numpy()


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
        max_batch_size = min(nominal_batch_size, int(max_size//(shape[1]*shape[2]*shape[3])))
        if max_batch_size < nominal_batch_size:
            print("  Warn: Batch too large. Subbatching.")
            for i in range(0, shape[0], max_batch_size):
                yield (a[i:i+max_batch_size].contiguous(), b[i:i+max_batch_size].contiguous(), c[i:i+max_batch_size].contiguous(), d[i:i+max_batch_size].contiguous())
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

class BaseLogger(object):
    def __init__(self):
        pass
    def log_epoch(self, epoch, loss_results, wer_results, cer_results, eval_loss=None):
        pass
    def log_previous_epochs(self, end_epoch, loss_results, wer_results, cer_results):
        pass

class VizLogger(BaseLogger):
    def __init__(self, _id, epochs):
        from visdom import Visdom
        self.viz = Visdom()
        self.opts = dict(title=_id, ylabel='', xlabel='Epoch', legend=['Loss', 'WER', 'CER'])
        self.viz_window = None
        self.epochs = torch.arange(1, epochs+1)

    def log_epoch(self, epoch, loss_results, wer_results, cer_results, eval_loss=None):
        x_axis = self.epochs[0:epoch + 1]
        y_axis = torch.stack((loss_results[0:epoch + 1], wer_results[0:epoch + 1], cer_results[0:epoch + 1]), dim=1)
        if self.viz_window is None:
            self.viz_window = viz.line(
                X=x_axis,
                Y=y_axis,
                opts=self.opts,
            )
        else:
            viz.line(
                X=x_axis.unsqueeze(0).expand(y_axis.size(1), x_axis.size(0)).transpose(0, 1),  # Visdom fix
                Y=y_axis,
                win=self.viz_window,
                update='replace',
            )
    def log_previous_epochs(self, end_epoch, loss_results, wer_results, cer_results):
        x_axis = epochs[0:end_epoch]
        y_axis = torch.stack(
            (loss_results[0:end_epoch], wer_results[0:end_epoch], cer_results[0:end_epoch]),
            dim=1)
        self.viz_window = viz.line(
            X=x_axis,
            Y=y_axis,
            opts=opts,
        )

class TensorboardLogger(BaseLogger):
    def __init__(self, _id, log_dir, model=None):
        from tensorboardX import SummaryWriter
        import socket
        from datetime import datetime
        log_dir = os.path.join(log_dir, datetime.now().strftime('%b%d_%H-%M-%S')+'_'+socket.gethostname()+'_'+_id)
        try:
            os.makedirs(log_dir)
        except OSError as e:
            if e.errno == errno.EEXIST:
                print('Tensorboard log directory already exists.')
                for file in os.listdir(log_dir):
                    file_path = os.path.join(log_dir, file)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                    except Exception:
                        raise
            else:
                raise
        self._writer = SummaryWriter(log_dir)
        self._id = _id
        self._model = model

    def log_epoch(self, epoch, loss_results, wer_results, cer_results, eval_loss=None):
        # values = {
        #     'Avg Train Loss': loss_results[epoch],
        #     'Avg WER': wer_results[epoch],
        #     'Avg CER': cer_results[epoch]
        # }
        # self._writer.add_scalars(self._id, values, epoch + 1)
        self._writer.add_scalar("loss/train", loss_results[epoch], epoch+1)
        self._writer.add_scalar("loss/val", eval_loss, epoch+1)
        self._writer.add_scalar("accuracy/wer", wer_results[epoch], epoch+1)
        self._writer.add_scalar("accuracy/cer", cer_results[epoch], epoch+1)
        if self._model:
            for tag, value in self._model.named_parameters():
                tag = tag.replace('.', '/')
                self._writer.add_histogram(tag, to_np(value), epoch + 1)
                self._writer.add_histogram(tag + '/grad', to_np(value.grad), epoch + 1)

    def log_previous_epochs(self, end_epoch, loss_results, wer_results, cer_results):
        for i in range(end_epoch):
            values = {
                'Avg Train Loss': loss_results[i],
                'Avg WER': wer_results[i],
                'Avg CER': cer_results[i]
            }
            self._writer.add_scalars(self._id, values, i + 1)


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
                      window=args.window,
                      noise_dir=args.noise_dir,
                      noise_prob=args.noise_prob,
                      noise_levels=(args.noise_min, args.noise_max))

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

        total_loss += (criterion(out, targets, sizes, target_sizes) / inputs.size(0))

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
    wer = (total_wer / len(test_loader.dataset)) * 100
    cer = (total_cer / len(test_loader.dataset)) * 100
    loss = (total_loss / len(test_loader.dataset))
    return wer, cer, loss

if __name__ == '__main__':
    torch.manual_seed(123456)
    torch.cuda.manual_seed_all(123456)
    args = parser.parse_args()

    save_folder = args.save_folder
    create_dir_safe(save_folder)

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
                                       normalize=True, augment=args.augment)
    test_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.val_manifest, labels=labels,
                                      normalize=True, augment=False)
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

    # iterate over dataset an epoch at a time
    for epoch in range(ts.start_epoch, args.epochs):
        model.train()
        end = time.time()
        for i, (data) in enumerate(train_loader, start=ts.start_iter):
            if i == len(train_sampler):
                break
            # measure data loading time
            data_time.update(time.time() - end)
            for inputs, targets, input_percentages, target_sizes in get_subbatches(data, args.batch_size, max_size=2000000):
                inputs = Variable(inputs, requires_grad=False)
                target_sizes = Variable(target_sizes, requires_grad=False)
                targets = Variable(targets, requires_grad=False)

                if args.cuda:
                    inputs = inputs.cuda()

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
                losses.update(loss_value, args.batch_size)

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
            if not args.silent:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    (epoch + 1), (i + 1), len(train_sampler), batch_time=batch_time,
                    data_time=data_time, loss=losses))
            if args.checkpoint_per_batch > 0 and i > 0 and (i + 1) % args.checkpoint_per_batch == 0:
                file_path = '%s/deepspeech_checkpoint_epoch_%d_iter_%d.pth.tar' % (save_folder, epoch + 1, i + 1)
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
            file_path = '%s/deepspeech_%d.pth.tar' % (save_folder, epoch + 1)
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

