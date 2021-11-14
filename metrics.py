import numpy as np


class MetricsHelper:
    """
    Metrics in the employed datasets require per-user computation due to memory constraints.
    This class helps manage intermediate metrics for computing hit rate (HR) and
    normalized discounted cumulative gain (NDCG) and additionally handles updating steps.

    Usage:
    Call update_hr_ndcg(...) with the given set of values to incorporate them into the metrics. Use the getter functions
    to obtain the final results.
    """

    def __init__(self, top_k):
        assert top_k is not None and len(top_k) > 0

        self.top_k = top_k
        self.total_buys, self.total_clicks = 0, 0
        self.hr_click, self.hr_buy = [0] * len(top_k), [0] * len(top_k)
        self.ndcg_click, self.ndcg_buy = [0] * len(top_k), [0] * len(top_k)

    def update_hr_ndcg(self, preds, actions, buys):
        """updates the metrics with the set of given values"""

        assert len(preds) == len(actions) == len(buys)  # all lists must be of equal lengths

        # update totals
        self.total_buys += sum(buys)
        self.total_clicks += len(buys) - sum(buys)

        # update metrics
        for idx_k, k in enumerate(self.top_k):
            top_k_preds = preds[:, 0:k]  # retrieve the top-k actions the model predicted
            for idx_a, gt in enumerate(actions):
                rank = np.argwhere(gt == top_k_preds[idx_a]).squeeze(axis=-1)  # retrieve correct item from pred. list
                if len(rank) > 0:  # match found, else item not predicted
                    assert len(rank) == 1
                    if buys[idx_a]:
                        self.hr_buy[idx_k] += 1
                        self.ndcg_buy[idx_k] += 1 / np.log2(rank[0] + 1 + 1)  # ndcg penalizes by rank
                    else:
                        self.hr_click[idx_k] += 1
                        self.ndcg_click[idx_k] += 1 / np.log2(rank[0] + 1 + 1)

    def get_hr_ndcg(self):
        """normalizes values by total count and returns them"""
        assert self.total_buys > 0 and self.total_clicks > 0
        hr_click, hr_buy = [i / self.total_clicks for i in self.hr_click], \
                           [i / self.total_buys for i in self.hr_buy]
        ndcg_click, ndcg_buy = [i / self.total_clicks for i in self.ndcg_click], \
                               [i / self.total_buys for i in self.ndcg_buy]
        return (hr_click, hr_buy), (ndcg_click, ndcg_buy)

    def get_hr(self):
        """normalizes values by total count and returns them"""
        assert self.total_clicks > 0 and self.total_buys > 0
        hr_click, hr_buy = [i / self.total_clicks for i in self.hr_click], \
                           [i / self.total_buys for i in self.hr_buy]
        return hr_click, hr_buy

    def get_ndcg(self):
        """normalizes values by total count and returns them"""
        assert self.total_buys > 0 and self.total_clicks > 0
        ndcg_click, ndcg_buy = [i / self.total_clicks for i in self.ndcg_click], \
                               [i / self.total_buys for i in self.ndcg_buy]
        return ndcg_click, ndcg_buy

    def get_click_metrics(self):
        """normalizes values by total count and returns them"""
        assert self.total_clicks > 0
        hr_click = [i / self.total_clicks for i in self.hr_click]
        ndcg_click = [i / self.total_clicks for i in self.ndcg_click]
        return hr_click, ndcg_click

    def get_buy_metrics(self):
        """normalizes values by total count and returns them"""
        assert self.total_buys > 0
        hr_buy = [i / self.total_buys for i in self.hr_buy]
        ndcg_buy = [i / self.total_buys for i in self.ndcg_buy]
        return hr_buy, ndcg_buy
