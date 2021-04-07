from tokenizers import BertWordPieceTokenizer
import argparse
import os
import tensorflow as tf
from pprint import pprint

"""
    python3 create_vocabs.py --corpus_dir ./corpus --output_dir ../cased
    or 
    python3 create_vocabs.py --corpus_dir ./corpus --output_dir ../uncased --uncased
"""

def parse_commandline():
    # Make parser object
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_files", required=True, help = "pattern for all files in corpus [may be splitted with ,]")
    parser.add_argument("--uncased", action="store_true", help="state whether tokenizer is cased")
    parser.add_argument("--output_dir", help="directory for vocab.txt")

    return (parser.parse_args())

if __name__ == "__main__":

    args = vars(parse_commandline())

    if not os.path.exists(args['output_dir']):
        os.makedirs(args['output_dir'])

    paths = []
    for file in args['input_files'].split(","):
        paths.extend(tf.gfile.Glob(file))

    # In this example we are using pretrained BERT Tokenizer to create vocab
    # So if it is necessary, think about training your own tokenizer
    tokenizer = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=False,
        strip_accents=True,
        lowercase=args['uncased'],
    )

    tokenizer.train(
        paths,
        vocab_size=32000,
        min_frequency=2,
        show_progress=True,
        special_tokens=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'],
        limit_alphabet=1000,
        wordpieces_prefix="##"
    )

    tokenizer.save(args['output_dir'], "")
