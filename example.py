from chesskurcz.algorithms.encoder import Encoder

enc = Encoder(alg='apm', num_workers=4, batch_size=1e5)
enc.encode(in_stream='<input file>', out_stream='<output file>')
enc.decode(in_stream='<encoded file>', out_stream='<output file>')