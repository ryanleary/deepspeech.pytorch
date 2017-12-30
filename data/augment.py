
class AudioReader(object):
    def __init__(self,):
        pass


def load_audio(path):
    sound, _ = torchaudio.load(path)
    sound = sound.numpy()
    if len(sound.shape) > 1:
        if sound.shape[1] == 1:
            sound = sound.squeeze()
        else:
            sound = sound.mean(axis=1)  # multiple channels, average
    return sound
