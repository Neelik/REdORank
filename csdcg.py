import numpy as np
from metrics import Scorer
from utils import group_offsets


# def _ncsdcg_score(y_true, y_pred, qid, feat, k=None, csdcg_func=None):
#     csdcg = np.array([csdcg_func(y_pred[a:b], feat[a:b], k=k)
#                       for a, b in group_offsets(qid)])
#     csdcg_best = np.array([csdcg_func(y_true[a:b], feat[a:b], k=k)
#                            for a, b in group_offsets(qid)])
#     csdcg_worst = np.array([csdcg_func(y_true[a:b], feat[a:b], k=k, worst=True)
#                             for a, b in group_offsets(qid)])
#     idcg = np.array([dcg_func(np.sort(y_true[a:b]), np.arange(0, b - a), k=k)
#                      for a, b in group_offsets(qid)])
#     assert (csdcg_worst <= csdcg <= csdcg_best).all()
#     ncsdcg = np.subtract(csdcg, csdcg_worst) / np.subtract(csdcg_best, csdcg_worst)
#     print(ncsdcg)
#     return ncsdcg
#
#
# def _allen_csdcg(y_pred, feat, mixer, k=None, worst=False):
#     if worst:
#         order = np.argsort(y_pred)
#     else:
#         order = np.argsort(-y_pred)
#     mixer_rel = mixer.predict(feat[order[:k], :2])
#     gain = 2 ** mixer_rel - 1
#     discounts = np.log2((order[:k] + 1) + 2)
#
#     return np.sum((gain / discounts) - feat[order[:k], 2])
#
#
# def _sayed_csdcg(y_true, y_pred, k=None, worst=False):
#     if worst:
#         order = np.argsort(y_pred)
#     else:
#         order = np.argsort(-y_pred)
#     # order = np.argsort(-y_pred)
#     y_true = np.take(y_true, order[:k])
#     gain = 2 ** y_true - 1
#     discounts = np.log2(np.arange(len(gain)) + 2)
#     return np.sum(gain / discounts)
#
#
# def _dcg_score(y_true, y_pred, qid, k=None, dcg_func=None):
#     assert dcg_func is not None
#     y_true = np.maximum(y_true, 0)
#     return np.array([dcg_func(y_true[a:b], y_pred[a:b], k=k) for a, b in group_offsets(qid)])
#
#
# def _csdcg_score(y_true, y_pred, qid, k=None, dcg_func=None):
#     assert dcg_func is not None
#     y_true = np.maximum(y_true, 0)
#     dcg = _dcg_score(y_true, y_pred, qid, k=k, dcg_func=dcg_func)
#     idcg = np.array([dcg_func(np.sort(y_true[a:b]), np.arange(0, b - a), k=k)
#                      for a, b in group_offsets(qid)])
#     assert (dcg <= idcg).all()
#     idcg[idcg == 0] = 1
#     return dcg / idcg
#
#
# def ncsdcg_score(y_true, y_pred, qid, feat, k=None, version='sayed'):
#     assert version in ['sayed', 'allen']
#     csdcg_func = _sayed_csdcg if version == 'sayed' else _allen_csdcg
#     return _ncsdcg_score(y_true, y_pred, qid, feat, k=k, csdcg_func=csdcg_func)
#
#
# class NCSDCGScorer(Scorer):
#     def __init__(self, **kwargs):
#         super(NCSDCGScorer, self).__init__(ncsdcg_score, **kwargs)
#
#     # def determine_bounds(self, x):


# EXPERIMENTAL
def cs_dcg_at_k(y_pred, k, cost):
    gain = np.exp2(y_pred[:k]) - 1
    discounts = np.log2(np.arange(len(y_pred[:k])) + 2)
    return np.sum((gain / discounts) - cost[:k])


def ncsdcg_at_k(y_true, y_predict, X, k, cost_index):
    # print("ncsdcg_at_k", X.shape)
    label_indices = np.flip(np.argsort(y_predict))
    csdcg_labels = np.take(y_true, label_indices)

    # TODO Add sorting for y_true because not all datasets have relevance in sorted order in data file
    sorted_true = np.flip(np.argsort(y_true))
    y_true_sorted = np.take(y_true, sorted_true)

    if cost_index:
        cost = X[label_indices, cost_index]
        cost_best = X[:, cost_index]
        cost_best = np.take(cost_best, sorted_true)
        cost_worst = np.flip(np.take(X[:, cost_index], sorted_true))
    # if X.shape[1] > 1:
    #     cost = X[label_indices, -1]
    #     cost_best = X[:, -1]
    #     cost_worst = np.flip(X[:, -1])
    else:
        cost, cost_best, cost_worst = np.array([0]), np.array([0]), np.array([0])

    wcsdcg = cs_dcg_at_k(np.flip(y_true_sorted), k, cost_worst)
    icsdcg = cs_dcg_at_k(y_true_sorted, k, cost_best)
    csdcg = cs_dcg_at_k(csdcg_labels, k, cost)

    ncsdcg = (csdcg - wcsdcg) / (icsdcg - wcsdcg)
    # print((csdcg - wcsdcg), (icsdcg - wcsdcg), ncsdcg)
    # if np.isnan(ncsdcg):
    #     print("ncsdcg")
    return ncsdcg


class NCSDCGScorer(object):
    def __init__(self, k=5, cost_index=None):
        self.k = k
        self.cost_index = cost_index

    def __call__(self, y_true, y_predict, X):
        return ncsdcg_at_k(y_true, y_predict, X, self.k, self.cost_index)


class NCSDCGScorer_qid:
    def __init__(self, k=5, cost_index=None):
        self.scorer = NCSDCGScorer(k, cost_index)
        self.k = k

    def __call__(self, y_true, y_predict, qid, X):
        return self.get_scores(y_true, y_predict, qid, X)

    def get_scores(self, true_list, pred_l, qid, x):
        scores = []
        # get group off-set
        prv_qid = qid[0]
        mark_index = [0]
        for itr, a in enumerate(qid):
            if a != prv_qid:
                mark_index.append(itr)
                prv_qid = a
        mark_index.append(qid.shape[0])
        for start, end in zip(mark_index, mark_index[1:]):
            scores.append(self.scorer(true_list[start:end], pred_l[start:end], x[start:end, :]))

        return np.array(scores)
