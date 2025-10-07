import numpy as np


class Base(object):
    def __init__(self, seed: int, dim_action: int, topK: int):
        self._seed = seed
        self._dim_action = dim_action
        self._topK = topK

    def initialise(self, batch_size: int, ini_mu_scale=0.0, ini_cov_scale=1.0):
        self._rng = np.random.RandomState(self._seed)

        # init weights
        _w = np.array([np.log((self._topK + 1) / i) for i in range(1, self._topK + 1)][::-1])
        _w = _w / np.sum(_w, axis=-1)  # normalise
        self._weights = np.tile(A=_w[None, :], reps=(batch_size, 1))  # b x topK

        # init parameters
        self._mu = ini_mu_scale * np.ones((batch_size, self._dim_action))
        self._cov = ini_cov_scale * np.ones((batch_size, self._dim_action))

    def _sample_from_gaussian(self, num_samples: int):
        """ Returns a matrix of batch x num_samples x dim """
        _mu = np.tile(A=self._mu[:, None, :], reps=(1, num_samples, 1))  # b x num_samples x dim
        _cov = np.tile(A=self._cov[:, None, :], reps=(1, num_samples, 1))  # b x num_samples x dim
        return _mu + self._rng.normal(size=(self._mu.shape[0], num_samples, self._dim_action)) * np.sqrt(_cov)

    def sample(self, num_samples: int, rescale_output=False, env_max_action=1.0):
        """ Returns a matrix of batch x num_samples x dim """
        samples = self._sample_from_gaussian(num_samples=num_samples).astype(np.float32)
        if rescale_output:
            samples = np.tanh(samples) * env_max_action
        return samples

    def update(self, elite_samples: np.ndarray):
        """ Update the distribution

        Args:
            elite_samples: batch x topK x dim_action
                ** Ascending order!!
        """
        pass


class CEM(Base):
    """ Batched Version of Cross-entropy method, as optimization of the action policy
    """

    def update(self, elite_samples: np.ndarray):
        # Simple update
        # self._mu = np.mean(elite_samples, axis=1)
        # self._cov = np.var(elite_samples, axis=1)

        # === Weighted sum: Better traj has the larger weight
        # Update cov
        z = (elite_samples - np.tile(A=self._mu[:, None, :], reps=(1, self._topK, 1)))  # b x topK x dim
        z = np.einsum("abc,abc->abc", z, z)  # b x topK x dim
        self._cov = np.einsum("abc,ab->ac", z, self._weights)  # b x topK x dim

        # Update mu: weighted sum over the elites
        self._mu = np.einsum("abc,ab->ac", elite_samples, self._weights)  # b x dim


def test():
    batch_size, num_samples, dim_action, topK = 3, 10, 5, 2
    cem = CEM(seed=123, dim_action=dim_action, topK=topK)
    cem.initialise(batch_size=batch_size)
    for _ in range(4):
        samples = cem.sample(num_samples=num_samples)  # b x num_samples x dim
        scores = np.sum(samples, axis=-1)  # b x num_samples
        idx = np.argpartition(a=scores, kth=-topK)[:, -topK:]  # b x k
        cem.update(elite_samples=np.take_along_axis(arr=samples, indices=idx[..., None], axis=1))


if __name__ == '__main__':
    test()
