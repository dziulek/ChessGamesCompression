# Chess Games compression techniques

![CI status](https://github.com/dziulek/ChessGamesCompression/actions/workflows/build.yml/badge.svg)

This package provides several techniques to compress chess games.

Motivated by the rapidly increasing number of chess games available, 
I proposed few algorithms to compress big chess databases. In addition,
useful IO operations are implemented to manage big files and use them 
as an input to deep learning algorithms.

To install repository go to the root directory and run
```shell
$ pip install .
```

Simple example of encoding and decoding a [*.pgn](https://en.wikipedia.org/wiki/Portable_Game_Notation) file

```python
from chesskurcz.algorithms.encoder import Encoder

enc = Encoder(alg='apm', num_workers=4, batch_size=1e5)
enc.encode(in_stream='<input file>', out_stream='<output file>')
enc.decode(in_stream='<encoded file>', out_stream='<output file>')
```

If you just want to read `N` games from an encoded file type
```python
enc.decode_batch_of_games(path='<encoded file>', out_stream='<output file>', N=100)
```
