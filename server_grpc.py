"""Server-end for the ASR demo."""
import os
import sys
import time
import random
import argparse
import functools
from time import gmtime, strftime
from concurrent import futures
import grpc
import struct
import cloud as speech
import numpy as np
import wave
from model import DeepSpeech
from decoder import BeamCTCDecoder, GreedyDecoder
from data.data_loader import SpectrogramParser
from torch.autograd import Variable
import threading
import collections

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


parser = argparse.ArgumentParser(description='DeepSpeech transcription')
parser.add_argument('--host_ip', default='localhost', type=str, help='server ip addr')
parser.add_argument('--host_port', default=44551, type=int, help='server ip addr')
parser.add_argument('--model_path', default='models/deepspeech_final.pth.tar',
                    help='Path to model file created by training')
parser.add_argument('--language_code', default='en-US', type=str, help='language code for the model being served')
parser.add_argument('--cuda', action="store_true", help='Use cuda to test model')
parser.add_argument('--decoder', default="greedy", choices=["greedy", "beam"], type=str, help="Decoder to use")
beam_args = parser.add_argument_group("Beam Decode Options", "Configurations options for the CTC Beam Search decoder")
beam_args.add_argument('--beam_width', default=10, type=int, help='Beam width to use')
beam_args.add_argument('--lm_path', default=None, type=str,
                       help='Path to an (optional) kenlm language model for use with beam search (req\'d with trie)')
beam_args.add_argument('--trie_path', default=None, type=str,
                       help='Path to an (optional) trie dictionary for use with beam search (req\'d with LM)')
beam_args.add_argument('--lm_alpha', default=1.5, type=float, help='Language model weight')
beam_args.add_argument('--lm_beta', default=0.03, type=float, help='Language model word bonus (all words)')
beam_args.add_argument('--label_size', default=0, type=int, help='Label selection size controls how many items in '
                                                                 'each beam are passed through to the beam scorer')
beam_args.add_argument('--label_margin', default=-1, type=float, help='Controls difference between minimal input score '
                                                                      'for an item to be passed to the beam scorer.')
args = parser.parse_args()


class CloudSpeechServicer(speech.SpeechServicer):
    def __init__(self, model, parser, decoder, language, min_buffer=1.0, max_buffer=3.0, flush_time=2.0):
        self._model = model
        self._parser = parser
        self._decoder = decoder
        self._model_sample_rate = DeepSpeech.get_audio_conf(self._model).get("sample_rate", 16000)
        self._flush_time = flush_time  # 100 ms
        self._max_buffer = int(max_buffer * self._model_sample_rate)  # 200 ms
        self._min_buffer = int(min_buffer * self._model_sample_rate) # 50 ms
        self._language = language


    def _raw_data_to_samples(self, data, sample_rate=16000, encoding=None):
        # TODO: support other encodings
        if sample_rate == 16000 and encoding == speech.RecognitionConfig.LINEAR16:
            signal = np.frombuffer(data, dtype=np.int16)
        else:
            raise ValueError("Unsupported audio data configuration")
            signal = None
        return signal

    def _get_np_from_deque(self, data, size=10, reserve=1000):
        out = np.zeros((size,))
        for x in range(size):
            out[x] = data.popleft()
        for x in range(reserve):
            data.appendleft(out[-(x+1)])
        return out

    def StreamingRecognize(self, request_iterator, context):
        print("Handling stream request...")
        config_wrapper = request_iterator.next()
        if not config_wrapper.HasField("streaming_config"):
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details('First StreamingRequest must be a configuration request')
            return
            # return an error
        stream_config = config_wrapper.streaming_config

        # check audio format (sample rate, encoding) to convert if necessary
        if stream_config.config.language_code != self._language:
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details('Requested unsupported language')
            return

        sample_buffer = collections.deque()
        done = False
        last_incoming = time.time()
        def read_incoming():
            try:
                while 1:
                    received = next(request_iterator)
                    samples = self._raw_data_to_samples(received.audio_content, sample_rate=stream_config.config.sample_rate_hertz, encoding=stream_config.config.encoding)
                    sample_buffer.extend(samples)
                    last_incoming = time.time()
            except StopIteration:
                print("reached end")
                return
            except ValueError:
                context.set_code(grpc.StatusCode.UNIMPLEMENTED)
                context.set_details('Unable to handle requested audio type')
                raise ValueError('Unable to handle requested audio type')

        thread = threading.Thread(target=read_incoming)
        thread.daemon = True
        thread.start()

        last_check = time.time()
        full_transcript = ""
        hidden = None
        result = None
        last_buffer_size = -1
        while 1:
            stream_done = time.time()-last_incoming > self._flush_time
            if len(sample_buffer) > self._min_buffer or (time.time()-last_check >= self._flush_time and len(sample_buffer) > self._min_buffer):
                last_check = time.time()
                signal = self._get_np_from_deque(sample_buffer, size=min(len(sample_buffer), self._max_buffer), reserve=int(0.4*self._model_sample_rate))
                spect = self._parser.parse_audio_data(signal).contiguous()
                spect = spect.view(1, 1, spect.size(0), spect.size(1))
                out, _ = self._model(Variable(spect, volatile=True), hidden)
                out = out.transpose(0, 1)  # TxNxH
                decoded_output, _, _, _ = self._decoder.decode(out.data[:-19,:,:])
                full_transcript += decoded_output[0][0]
                alt = speech.SpeechRecognitionAlternative(transcript=full_transcript)
                result = speech.StreamingRecognitionResult(alternatives=[alt], is_final=done)
                out = speech.StreamingRecognizeResponse(results=[result])
                # if stream_done:
                #     return out
                yield out
            else:
                last_check = time.time()
                time.sleep(0.01)

def serve():
    model = DeepSpeech.load_model(args.model_path, cuda=args.cuda)
    model.eval()

    labels = DeepSpeech.get_labels(model)
    audio_conf = DeepSpeech.get_audio_conf(model)

    if args.decoder == "beam":
        from decoder import BeamCTCDecoder

        decoder = BeamCTCDecoder(labels, beam_width=args.beam_width, top_paths=1,
                                 space_index=labels.index(' '),
                                 blank_index=labels.index('_'), lm_path=args.lm_path,
                                 trie_path=args.trie_path, lm_alpha=args.lm_alpha, lm_beta=args.lm_beta,
                                 label_size=args.label_size, label_margin=args.label_margin)
    else:
        decoder = GreedyDecoder(labels, space_index=labels.index(' '), blank_index=labels.index('_'))

    parser = SpectrogramParser(audio_conf, normalize=True)

    print("Model loaded.")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    speech.cloud_speech_pb2_grpc.add_SpeechServicer_to_server(CloudSpeechServicer(model, parser, decoder, args.language_code), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server started.")

    while True:
        try:
            time.sleep(_ONE_DAY_IN_SECONDS)
        except (KeyboardInterrupt):
            server.stop(0)
            sys.exit(1)

if __name__ == "__main__":
    serve()
