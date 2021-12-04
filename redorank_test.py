"""
File to test different AdaRank repositories to find one suited for my thesis work
"""
# Shared imports
from __future__ import print_function, division
import numpy as np
from pathlib import Path

# AdaRank imports
from adarank import AdaRank as AR
# from cs_dcg import NDCGScorer_qid
from metrics import NDCGScorer
from read_data import read_data


def cal():
    classes = [AR]
    for cls in classes:
        print("Loading data")
        k_max = 10
        folds = 1
        data_path = Path(__file__).resolve().parent.joinpath("datasets", "mslr")
        parser = read_data()
        parser.read_mq2008(data_path, folds=folds)
        # parser.read_ml()
        scores = []

        for k in range(k_max):
            scores.append([])

        for i in range(folds):
            print("============fold{}==================".format(i + 1))
            train, vali, test = parser.get_fold(i)
            x, y, qid = train

            x_test, y_test, qid_test = test
            x_vali, y_vali, qid_vali = vali

            print("Training model")
            model = cls(scorer=NDCGScorer(k=k_max))
            model.fit(x, y, qid, x_vali, y_vali, qid_vali)

            pred = model.predict(x_test, qid_test)
            for k in range(k_max):
                score = round(NDCGScorer(k=k + 1)(y_test, pred, qid_test).mean(), 4)
                scores[k].append(score)
                print('nDCG@{}\t{}\n'.format(k + 1, score))
        print("==============Mean NDCG==================")
        for f in range(k_max):
            print("mean NDCG@{}\t\t{}\n".format(f + 1, round(np.mean(scores[f]), 4)))


if __name__ == "__main__":
    print("Running test")
    cal()
