# -*- coding: utf-8 -*-

from tokenizers import BertWordPieceTokenizer
import argparse
import os
import tensorflow as tf
from pprint import pprint

"""
    python3 create_vocabs.py --input_file ./turkishcorpus  --output_dir ./cased
    or 
    python3 create_vocabs.py --input_file ./turkishcorpus --output_dir ./uncased --uncased
"""

def parse_commandline():
    # Make parser object
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", required=True, help = "Input files (can be a glob or comma separated).")
    parser.add_argument("--uncased", action="store_true", help="state whether tokenizer is cased")
    parser.add_argument("--output_dir", help="directory for vocab.txt")

    return (parser.parse_args())

if __name__ == "__main__":

    args = vars(parse_commandline())

    if not os.path.exists(args['output_dir']):
        os.makedirs(args['output_dir'])

    paths = []
    fileprefix = 'uncased' if args['uncased'] else 'cased'
    for pattern in args['input_file'].split(","):
        paths.extend(tf.io.gfile.glob(pattern))

    # In this example we are using pretrained BERT Tokenizer to create vocab
    # So if it is necessary, think about training your own tokenizer
    tokenizer = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=False,
        strip_accents=False,
        lowercase=args['uncased'],
    )

    tokenizer.train(
        files=paths,
        vocab_size=32000,
        min_frequency=2,
        show_progress=True,
        special_tokens=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'],
        limit_alphabet=1000,
        wordpieces_prefix="##"
    )

    tokenizer.save(args['output_dir'], fileprefix)
