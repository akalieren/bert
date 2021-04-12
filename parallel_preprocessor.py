import subprocess
import shlex
import os
import argparse

"""
python3 bert/parallel_preprocessor.py --script bert/create_pretraining_data.py --corpus /home/akali/data/oscar/corpus_splitted/ --output ./tfrecords --max_seq_length 512 --num_thread 3  --vocab_file ./vocabs/uncased/-vocab.txt --uncased
"""

def parse_commandline():
    # Make parser object
    parser = argparse.ArgumentParser()

    parser.add_argument("--script", required=True, help = "path to create_pretraining.py")
    parser.add_argument("--corpus", required=True, help="corpus path to prepare for training")
    parser.add_argument("--output", required=True, help="output file for TFRecords")
    parser.add_argument("--max_seq_length", default=512, help="tokenizers' max seq length. Default is 512")
    parser.add_argument("--uncased", action='store_true', help="do lower case flag for tokenizers")
    parser.add_argument("--vocab_file", required=True, help="Vocab file for tokenizers")
    parser.add_argument("--num_thread", default=5, help="Number of workers for this task. This scripts consumes huge energy so consider your MEMORY size for this flag")
    parser.add_argument("--extra_arguments", default='', help="Extra tuning parameters for model masked_lm prob etc. refer scripts itself")

    return (parser.parse_args())

if __name__ == "__main__":
    args = vars(parse_commandline())

    train = os.listdir(args['corpus'])  # data splits

    for i in range(0, len(train), int(args['num_thread'])):  # 20 processes at a time
        subs = []
        for filename in train[i:i + int(args['num_thread'])]:
            command = f"python3 {args['script']} --input_file={os.path.join(args['corpus'], filename)} --output_file={args['output']} --mask=True --max_seq_length={args['max_seq_length']} --vocab_file={args['vocab_file']} --do_lower_case={args['uncased']} {args['extra_arguments']}"
            subs.append(subprocess.Popen(shlex.split(command)))

        for p in subs:
            p.communicate()  # sync process before starting another batch
