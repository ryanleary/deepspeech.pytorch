"""Google Cloud Speech API sample application using the streaming API.
Example usage:
    python transcribe_streaming.py resources/audio.raw
"""

# [START import_libraries]
import argparse
import io
import grpc
import cloud as speech
# [END import_libraries]


# [START def_transcribe_streaming]
def transcribe_streaming(stream_file):
    """Streams transcription of the given audio file."""
    channel = grpc.insecure_channel('localhost:50051')
    client = speech.SpeechStub(channel)
    #client = speech.SpeechClient()

    # [START migration_streaming_request]
    with io.open(stream_file, 'rb') as audio_file:
        content = audio_file.read()

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.LINEAR16,
        sample_rate_hertz=16000,
        language_code='en-US')
    streaming_config = speech.StreamingRecognizeRequest(streaming_config=speech.StreamingRecognitionConfig(config=config))
    # In practice, stream should be a generator yielding chunks of audio data.
    stream = [streaming_config] +  [speech.StreamingRecognizeRequest(audio_content=content[i:i+4]) for i in range(0,len(content),4)]
    requests = (chunk for chunk in stream)

    print(content[0])
    print(content[len(content)-1])


    # streaming_recognize returns a generator.
    # [START migration_streaming_response]
    responses = client.StreamingRecognize(requests)
    # [END migration_streaming_request]

    for response in responses:
        print("response received")
        for result in response.results:
            print('Finished: {}'.format(result.is_final))
            print('Stability: {}'.format(result.stability))
            alternatives = result.alternatives
            for alternative in alternatives:
                print('Confidence: {}'.format(alternative.confidence))
                print('Transcript: {}'.format(alternative.transcript))
    # [END migration_streaming_response]
# [END def_transcribe_streaming]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stream', help='File to stream to the API')
    args = parser.parse_args()
    transcribe_streaming(args.stream)
