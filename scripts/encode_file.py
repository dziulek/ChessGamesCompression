import os
import time
import shutil
from pathlib import Path
import argparse
import gzip

from chesskurcz.algorithms.encoder import Encoder
from chesskurcz.algorithms.util.utils import time_elapsed

@time_elapsed()
def encode_dir(input_path: Path, output_path: Path, 
                  encoder: Encoder, suffix: str, zip_alg='gzip', **kwargs) -> None:
    assert input_path.is_dir() == output_path.is_dir() 
    if input_path.is_dir():
        output_path.mkdir(exist_ok=True)
        for path in input_path.iterdir():
            encode_dir(path, output_path / path.parts[-1], encoder=encoder, suffix=suffix)
    else: 
        output_file_name = output_path.parts[-1].split('.')[0]
        output_file_name = f"{output_file_name}_{suffix}.bin"
        path = output_path.parent / output_file_name
        try:
            encoder.encode(str(input_path), str(path), **kwargs)
            compress_file(path, path, zip_alg)
        except Exception as e:
            print(f"[ERROR] Exception raised when encoding file: {str(input_path)}.")
        # os.remove(str(path))

def compress_file(path: Path, output_path: Path, alg='gzip'):

    if alg.lower() == 'gzip':
        with path.open('rb') as f_in:
            with gzip.open(f"{str(output_path)}.gz", 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
         
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", type=str)
    parser.add_argument("--algorithm", "-a", type=str, default='apm')
    parser.add_argument("--read-size", "-r", type=int, default=8192)
    parser.add_argument("--suffix", "-s", type=str, default='enc')
    parser.add_argument("--num-workers", "-n", type=int, default=1)
    parser.add_argument("--output-folder", "-o", type=str, default=None)
    parser.add_argument("--compress-output", "-c", action="store_true")
    parser.add_argument("--compression-algorithm", '-ca', type=str, default='gzip')
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--collect-stats", "-t", action='store_true')

    args = parser.parse_args()
    
    input_path = Path(os.path.join(os.getcwd(), args.path))
    if input_path.is_dir():
        output_path = Path(os.path.join(os.getcwd(), args.output_folder)) if args.output_folder is not None else input_path
    else:
        output_path = input_path

    encoder = Encoder(
        alg=args.algorithm,
        par_workers=args.num_workers,
        batch_size=args.read_size
    )

    duration = encode_dir(input_path, output_path, encoder, 
                        #   collect_stats=args.collect_stats,
                          suffix=args.suffix, 
                          verbose=args.verbose)
    print(f"Time:{duration}")
    # print(f"Avg.Compression Ratio:{}}")

if __name__ == "__main__":
    main()