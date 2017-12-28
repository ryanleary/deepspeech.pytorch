import torch


def to_np(x):
    return x.data.cpu().numpy()

class BaseLogger(object):
    def __init__(self):
        pass
    def init_epoch(self, loss, wer, cer, eval_loss):
        pass
    def log_epoch(self, epoch, loss_results, wer_results, cer_results, eval_loss=None):
        pass
    def log_previous_epochs(self, end_epoch, loss_results, wer_results, cer_results):
        pass
    def log_step(self, step, loss):
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

    def log_step(self, step, loss):
        self._writer.add_scalar("loss_step/train", loss, step+1)

    def init_epoch(self, loss, wer, cer, eval_loss):
        self._writer.add_scalar("loss/train", loss, 0)
        self._writer.add_scalar("loss/val", eval_loss, 0)
        self._writer.add_scalar("accuracy/wer", wer, 0)
        self._writer.add_scalar("accuracy/cer", cer, 0)

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
