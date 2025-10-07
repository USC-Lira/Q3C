import numpy as np

from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler  # this is to clearly constraint the action space


class Distribution(object):
    """Supposed to be used in DocumentSampler to instantiate the doc features."""

    def __init__(self, args: dict):
        self._args = args
        self._rng = np.random.RandomState(self._args["env_seed"])
        self._centroids = np.eye(self._args["recsim_dim_embed"])
        self._cov_mat = np.eye(self._args["recsim_dim_embed"]) * 0.05  # empirically determined

        if self._args["recsim_item_dist"].lower() == "sklearn-gmm":
            embedding, y = make_blobs(n_samples=self._args["num_all_actions"],
                                      # centers=self._args["recsim_num_categories"],
                                      centers=self._args["sklearn_gmm_centroids"],
                                      n_features=self._args["recsim_dim_embed"],
                                      cluster_std=0.35,
                                      # cluster_std=0.75,
                                      center_box=(-1.0, 1.0),
                                      random_state=self._args["env_seed"])
            embedding = MinMaxScaler(feature_range=(-1.0, 1.0)).fit_transform(embedding)
            self._embed = embedding
            self._category_arr = y

    def reset_sampler(self):
        self._rng = np.random.RandomState(self._args["env_seed"])

    def sample_item_embed_and_category(self, item_id: int, category_id: int = None, flg: str = None):
        """ MultiUserWrapper calls this to sample global item embedding """
        # === Sample Item Category
        if category_id is None:
            if self._args["recsim_category_dist"] == "random":
                category_id = self._rng.randint(low=0, high=self._args["recsim_num_categories"])
            else:
                raise ValueError

        # === Sample Item embedding
        if flg is None:
            if self._args["recsim_item_dist"].lower() == "simple-gmm":
                embed = np.zeros(self._args["recsim_dim_embed"])
                embed[category_id] = self._rng.uniform(low=0.6, high=1.0, size=1)
            elif self._args["recsim_item_dist"].lower() == "gmm":
                embed = self._rng.multivariate_normal(mean=self._centroids[category_id], cov=self._cov_mat, size=1)[0]
            elif self._args["recsim_item_dist"].lower() == "sklearn-gmm":
                embed = self._embed[item_id, :]
                category_id = self._category_arr[item_id]
            elif self._args["recsim_item_dist"].lower() == "pos-gmm":  # positive region biased gmm
                embed = self._rng.uniform(low=-1.0, high=0.2, size=self._args["recsim_dim_embed"])
                embed[category_id] = self._rng.uniform(low=0.5, high=1.0, size=1)
            elif self._args["recsim_item_dist"].lower() == "two-modal-gmm":
                assert self._args["recsim_num_categories"] == 2
                if item_id == 0:  # category_id == 0 and top right corner
                    embed = np.asarray([1.0, 1.0])
                    category_id = 0
                elif item_id == 1:  # category_id == 0 and bottom left corner
                    embed = np.asarray([-1.0, -1.0])
                    category_id = 0
                elif item_id == 2:  # category_id == 1 and top left corner
                    embed = np.asarray([1.0, -1.0])
                    category_id = 1
                elif item_id == 3:  # category_id == 1 and bottom right corner
                    embed = np.asarray([-1.0, 1.0])
                    category_id = 1
            else:
                raise ValueError
        else:
            if self._args["recsim_item_dist"].lower() == "two-head-uniform":
                if flg == "train":
                    embed = self._rng.uniform(low=-0.6, high=0.6, size=self._args["recsim_dim_embed"])
                elif flg == "test":
                    # Uniformly sample from two ranges
                    left_or_right = self._rng.random() > 0.5
                    if left_or_right:
                        embed = self._rng.uniform(low=-1.0, high=-0.6, size=self._args["recsim_dim_embed"])
                    else:
                        embed = self._rng.uniform(low=0.6, high=1.0, size=self._args["recsim_dim_embed"])
            else:
                raise ValueError
        return embed, category_id

    @classmethod
    def sample_user_interest(cls, rng: np.random.RandomState, args: dict):
        """ Samples individual user interest vector and Raw Env calls this to sample! Not MultiUserWrapper!! """
        if args["recsim_user_dist"].lower() == "uniform":
            embed = rng.uniform(low=-1.0, high=1.0, size=args["recsim_dim_embed"])
        elif args["recsim_user_dist"].lower() == "gmm":  # Same code as above
            _centroids = np.eye(args["recsim_dim_embed"])
            _cov_mat = np.eye(args["recsim_dim_embed"]) * 0.05  # empirically determined
            _category_id = rng.randint(low=0, high=args["recsim_num_categories"], size=1)[0]
            embed = rng.multivariate_normal(mean=_centroids[_category_id], cov=_cov_mat, size=1)[0]
        elif args["recsim_user_dist"].lower() == "sklearn-gmm":
            embed, _category_id = make_blobs(n_samples=1,
                                             centers=args["sklearn_gmm_centroids"],
                                             n_features=args["recsim_dim_embed"],
                                             cluster_std=0.35,
                                             center_box=(-1.0, 1.0),
                                             random_state=args["env_seed"])
            embed = embed[0]
        elif args["recsim_user_dist"].lower() == "modal":
            embed = np.zeros(args["recsim_dim_embed"])
            _category_id = rng.randint(low=0, high=args["recsim_num_categories"], size=1)
            embed[_category_id] = rng.uniform(low=0.8, high=1.0)
            # _vec[_category_id] = rng.uniform(low=1.0, high=5.0)
        else:
            raise ValueError

        if args["DEBUG_size_action_space"] != "unbounded":  # temp: row-wise min-max scaling
            embed = np.clip(a=embed, a_min=-1.0, a_max=1.0)
        return embed, _category_id


def test():
    print("=== test ===")

    # hyper-params
    args = {
        "recsim_if_varying_action_set": False,
        "recsim_num_categories": 10,
        "recsim_rejectionSampling_distance": 0.2,
        "num_allItems": 1000,
        "num_trainItems": 500,
        "num_testItems": 500,
    }

    for recsim_itemFeat_samplingMethod, recsim_itemFeat_samplingDist in [
        ("normal", None),
    ]:
        print(f"=== recsim_itemFeat_samplingMethod: {recsim_itemFeat_samplingMethod}, "
              f"recsim_itemFeat_samplingDist: {recsim_itemFeat_samplingDist}")
        args["recsim_itemFeat_samplingMethod"] = recsim_itemFeat_samplingMethod
        args["recsim_itemFeat_samplingDist"] = recsim_itemFeat_samplingDist

        dist = Distribution(args=args)

        print("=== Test: user sampling method ===")

        for train_test in [
            "train",
            "test"
        ]:
            print(train_test)
            for i in range(args[f"num_{train_test}Items"]):
                doc_feature = dist.sample(flg=train_test, item_id=i)
                print(f"item_id: {i}, feature: {doc_feature}")


if __name__ == '__main__':
    test()
