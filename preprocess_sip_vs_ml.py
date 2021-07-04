import pathlib
from argparse import ArgumentParser
import common
from preprocess import process_file


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-fp", "--file_path", dest="file_path", required=True, type=str)
    parser.add_argument("-mc", "--max_contexts", dest="max_contexts", default=200,
                        help="number of max contexts to keep", required=False)
    parser.add_argument("-wvs", "--word_vocab_size", dest="word_vocab_size", default=1301136,
                        help="Max number of origin word in to keep in the vocabulary", required=False)
    parser.add_argument("-pvs", "--path_vocab_size", dest="path_vocab_size", default=911417,
                        help="Max number of paths to keep in the vocabulary", required=False)
    parser.add_argument("-tvs", "--target_vocab_size", dest="target_vocab_size", default=261245,
                        help="Max number of target words to keep in the vocabulary", required=False)
    parser.add_argument("-wh", "--word_histogram", dest="word_histogram",
                        help="word histogram file", metavar="FILE", required=True)
    parser.add_argument("-ph", "--path_histogram", dest="path_histogram",
                        help="path_histogram file", metavar="FILE", required=True)
    parser.add_argument("-th", "--target_histogram", dest="target_histogram",
                        help="target histogram file", metavar="FILE", required=True)
    args = parser.parse_args()

    word_histogram_path = args.word_histogram
    path_histogram_path = args.path_histogram

    word_histogram_data = common.common.load_vocab_from_histogram(word_histogram_path, start_from=1,
                                                                  max_size=int(args.word_vocab_size),
                                                                  return_counts=True)
    _, _, _, word_to_count = word_histogram_data
    _, _, _, path_to_count = common.common.load_vocab_from_histogram(path_histogram_path, start_from=1,
                                                                     max_size=int(args.path_vocab_size),
                                                                     return_counts=True)
    _, _, _, target_to_count = common.common.load_vocab_from_histogram(args.target_histogram, start_from=1,
                                                                       max_size=int(args.target_vocab_size),
                                                                       return_counts=True)
    file_path = pathlib.Path(args.file_path)
    num_examples = process_file(
        file_path=str(file_path), data_file_role='preprocessed', dataset_name=file_path.name.split('.')[0],
        word_to_count=word_to_count, path_to_count=path_to_count, max_contexts=int(args.max_contexts)
    )
