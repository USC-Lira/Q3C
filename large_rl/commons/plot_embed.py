import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import OrderedDict
from functools import partial
from sklearn import manifold

NUM_DIM = 2


def plot_embedding(embedding: np.ndarray,
                   label_dict: dict,
                   if_all: bool = False,
                   title: str = None,
                   file_name: str = "",
                   save_dir: str = "./images",
                   num_neighbours: int = 2,
                   if_output_pdf: bool = False):
    # Create a dir if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    LLE = partial(manifold.LocallyLinearEmbedding, num_neighbours, NUM_DIM, eigen_solver="dense")
    model_dict = OrderedDict()
    if if_all:
        model_dict["LLE"] = LLE(method='standard')
        model_dict["LTSA"] = LLE(method='ltsa')
        model_dict["Hessian LLE"] = LLE(method='hessian')
        model_dict["Modified LLE"] = LLE(method='modified')
        model_dict["Isomap"] = manifold.Isomap(n_neighbors=num_neighbours, n_components=NUM_DIM)
        # methods["MDS"] = manifold.MDS(n_neighbors=NUM_NEIGHBOURS, max_iter=100, n_init=1)
        model_dict["SE"] = manifold.SpectralEmbedding(n_components=NUM_DIM, n_neighbors=num_neighbours)
        model_dict["t-SNE"] = manifold.TSNE(n_components=NUM_DIM, init='pca', random_state=0)
    else:
        model_dict["t-SNE"] = manifold.TSNE(n_components=NUM_DIM, init='pca', random_state=0, n_iter=1000)

    for _name, _model in model_dict.items():
        # print("=== {} ===".format(_name))
        file_name = "{}_{}".format(file_name, _name)
        _plot(_embedding=embedding, label_dict=label_dict, model=_model, file_name=file_name, save_dir=save_dir,
              title=title, if_output_pdf=if_output_pdf)


def _plot(_embedding, label_dict, model, file_name=None, title=None, save_dir="./images", if_output_pdf=False):
    # w = 2
    # h = w * 1.52
    # plt.figure(figsize=(h, w))

    if title is not None:
        plt.title(title)

    if _embedding.shape[-1] != 2:
        # Applying the model to transform the data
        Y = model.fit_transform(_embedding)
    else:
        Y = _embedding  # just plot as it is

    # List of colors in the color palettes
    # rgb_values = sns.color_palette("Set2", len(label_dict["desc"]))
    rgb_values = sns.hls_palette(n_colors=len(label_dict["desc"]), l=0.6, s=.9, h=0.7)

    # Map continents to the colors
    color_map = dict(zip(sorted(label_dict["desc"]), rgb_values))

    # Plotting the values with the labels and its predefined descriptions
    for _name, _value in sorted(label_dict["desc"].items()):
        mask = np.asarray(label_dict["labels"]) == _value
        if _name == "a":
            _marker = "x"
        elif _name == "s":
            _marker = "^"
        else:
            _marker = "."
        plt.scatter(x=Y[mask, 0], y=Y[mask, 1], c=[color_map[_name] for _ in range(sum(mask))], label=_name,
                    marker=_marker)

    plt.legend(loc=(1.04, 0), fontsize=10)
    # plt.legend(loc=2, fontsize=10)
    # plt.axis('tight')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.savefig(os.path.join(save_dir, "{}.png".format(file_name)), bbox_inches="tight")
    if if_output_pdf:
        plt.savefig(os.path.join(save_dir, "{}.pdf".format(file_name)), format="pdf", dpi=1000, bbox_inches="tight")
    plt.clf()
    plt.close()


def _test():
    from large_rl.commons.seeds import set_randomSeed
    set_randomSeed(seed=2022)
    num_samples = 50
    dim_sample = 10

    embedding = np.random.rand(num_samples, dim_sample)

    # Get the labels
    label_dict = dict()
    label_dict["labels"] = (np.random.uniform(size=num_samples) > 0.5).astype(np.int)
    label_dict["desc"] = {9: 1, 4: 0, }

    # plot_embedding(embedding=embedding, label_dict=label_dict, save_dir="/tmp/test-analysis/images", file_name="demo")
    plot_embedding(embedding=embedding, label_dict=label_dict, save_dir="./images", file_name="demo")


def _test2():
    from large_rl.commons.seeds import set_randomSeed
    from sklearn.datasets import make_blobs
    from sklearn.preprocessing import MinMaxScaler  # this is to clearly constraint the action space

    set_randomSeed(seed=2022)
    num_samples = 300
    dim_sample = 10

    embedding, y = make_blobs(n_samples=num_samples,
                              centers=3,
                              n_features=dim_sample,
                              cluster_std=0.35,
                              center_box=(-1.0, 1.0),
                              random_state=0)
    embedding = MinMaxScaler(feature_range=(-1, 1)).fit_transform(embedding)

    # print(embedding)
    print(embedding.max(), embedding.min())

    # Get the labels
    label_dict = dict()
    label_dict["labels"] = y
    label_dict["desc"] = {v: v for v in np.unique(y)}

    plot_embedding(embedding=embedding, label_dict=label_dict, save_dir="./images", file_name="demo")


if __name__ == '__main__':
    # _test()
    _test2()
