"""
    Evaluate on test data:
    python scorer.py --test_pred predictions.csv --test_label hidden.csv --test_result_file test_results.json

"""

import argparse
import json
import numpy as np
from sklearn.metrics import label_ranking_average_precision_score as LRAP


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_pred', help='path to prediction of test set csv file')
    parser.add_argument('--test_label', help='csv file containing the true labels for test set')
    parser.add_argument('--test_result_file', help='path to test set evaluation result file')
    args = parser.parse_args()
    return args


def main(args):
    test_pred =  np.genfromtxt(args.test_pred, delimiter=',')
    test_label =  np.genfromtxt(args.test_label, delimiter=',')

    lrap = LRAP(test_label, test_pred)
    print('test LRAP: ', lrap)
    with open(args.test_result_file, 'w') as fout:
        json.dump({'test LRAP': lrap}, fout)


if __name__ == '__main__':
    args = parse_args()
    main(args)
