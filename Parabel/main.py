"""
    Train model for evaluation on dev set (trained on training set only):
    python main.py --train_data data/train.txt --model_file demo_model

    Evaluate:
    python main.py --dev_data data/dev.csv --model_file demo_model --dev_result_file dev_results.json --evaluate

    Final training model for predicting test set (trained on training & dev set):
    python main.py --train_data data/train_dev.txt --model_file final_model

    Predict:
    python main.py --pred_data data/test_no_label.csv --model_file final_model --output_file predictions.csv --predict
"""

import argparse
import json
import os
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import label_ranking_average_precision_score as LRAP
import omikuji # Pakage implements Parabel Algorithm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', help='path to training data text file')
    parser.add_argument('--dev_data', help='path to dev data for evaluation')
    parser.add_argument('--dev_result_file', help='path to dev set evaluation result file')
    parser.add_argument('--pred_data', help='path to predict data csv file without label')
    parser.add_argument('--model_file', help='path to model file')
    parser.add_argument('--output_file', help='path to prediction output file')
    parser.add_argument('--evaluate', action='store_true', help='evaluate mode')
    parser.add_argument('--predict', action='store_true', help='predict mode')
    args = parser.parse_args()
    return args


def train(train_file, model_file):
    # Best Parabel parameters based on grid search
    # trained on the train set and evaluated on the dev set
    hyper_param = omikuji.Model.default_hyper_param()
    hyper_param.n_trees = 3 # default 3
    hyper_param.min_branch_size = 1000 # default 100
    hyper_param.linear_c = 1.2 # default 1.0
    hyper_param.max_depth = 50 # default 20

    # Train Parabel Model
    start_time = time.time()
    model = omikuji.Model.train_on_data(train_file, hyper_param)
    end_time = time.time()
    print('Training time: {}s'.format(end_time - start_time))

    # Save Parabel model
    model.save(model_file)
    print('Model saved to {}'.format(model_file))


def evaluate(dev_data, model_file, dev_result_file):
    # load data for prediction from csv
    data = pd.read_csv(dev_data, usecols=['features','labels'])

    # remove rows with improper label
    rows_to_remove = [i for i in range(len(data)) if ':' in data.loc[i,'labels']]
    data.drop(rows_to_remove, inplace=True)
    data.reset_index(drop=True, inplace=True)

    # Extract features from sparse representation
    feature = np.zeros((len(data), 5000))
    for i in range(len(data)):
        for j in data.loc[i,'features'].replace('\n','').split():
            ft, val = j.split(':')
            feature[i,int(ft)] = float(val)
    X = pd.DataFrame(feature)
    y = data['labels'].map(lambda x: tuple([int(i) for i in x.replace(' ','').split(',')]))
    binarizer = MultiLabelBinarizer(np.arange(3993))
    binary_y = binarizer.fit_transform(y)

    # Load saved model
    model = omikuji.Model.load(model_file)

    # Predict
    pred = np.zeros((X.shape[0], 3993))
    for i in range(X.shape[0]):
        feature_value_pairs = [(j, X.iloc[i,j]) for j in range(X.shape[1])]
        label_score_pairs = model.predict(feature_value_pairs, top_k=3993)
        for pairs in label_score_pairs:
            pred[i, pairs[0]] = pairs[1]

    # Calculate LRAP score
    lrap = LRAP(binary_y, pred)
    print('dev LRAP: ', lrap)
    with open(args.dev_result_file, 'w') as fout:
        json.dump({'dev LRAP': lrap}, fout)


def predict(pred_file, model_file, output_file):
    # load data for prediction from csv
    data = pd.read_csv(pred_file, usecols=['features'])
    data.reset_index(drop=True, inplace=True)

    # Extract features from sparse representation
    feature = np.zeros((len(data), 5000))
    for i in range(len(data)):
        for j in data.loc[i,'features'].replace('\n','').split():
            ft, val = j.split(':')
            feature[i,int(ft)] = float(val)
    X = pd.DataFrame(feature)

    # Load saved model
    model = omikuji.Model.load(model_file)

    # Predict
    pred = np.zeros((X.shape[0], 3993))
    for i in range(X.shape[0]):
        feature_value_pairs = [(j, X.iloc[i,j]) for j in range(X.shape[1])]
        label_score_pairs = model.predict(feature_value_pairs, top_k=3993)
        for pairs in label_score_pairs:
            pred[i, pairs[0]] = pairs[1]

    # Save prediction result to csv
    np.savetxt(output_file, pred, delimiter=",")
    print('Predictions saved to {}'.format(output_file))


def main(args):
    if args.evaluate:
        evaluate(args.dev_data, args.model_file, args.dev_result_file)

    elif args.predict:
        predict(args.pred_data, args.model_file, args.output_file)

    else:
        train(args.train_data, args.model_file)

if __name__ == '__main__':
    args = parse_args()
    main(args)
