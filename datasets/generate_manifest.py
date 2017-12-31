import argparse
from utils import create_manifest

parser = argparse.ArgumentParser(description='Processes and downloads LibriSpeech dataset.')
parser.add_argument("--data-dir", required=True, type=str, help="Directory containing txt and wav directories")
parser.add_argument('--output-file', required=True, type=str, help='Output manifest file')
parser.add_argument('--min-duration', default=None, type=int, required=False,
                    help='Prunes training samples shorter than the min duration (given in seconds, default None)')
parser.add_argument('--max-duration', default=None, type=int, required=False,
                    help='Prunes training samples longer than the max duration (given in seconds, default None)')
args = parser.parse_args()

if __name__ == '__main__':
    create_manifest(args.data_dir, args.output_file, min_duration=args.min_duration, max_duration=args.max_duration)
