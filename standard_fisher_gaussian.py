import numpy as np



class StandardFisherGaussian:

    def __init__(self, p, sigma2):
        self.p = p
        self.sigma2 = sigma2

    # evaluate density
    def pdf(x):
        # TODO: write down density function
        pass

    # sample
    def sample(self, n):

        # Sample from uniform sphere
        from scipy.stats import multivariate_normal as mvn
        gaussian_samples = mvn.rvs(mean=np.zeros(self.p), cov=1, size=n)

        # Normalize by L2 norm of each sample
        norms = np.linalg.norm(gaussian_samples, ord=2, axis=1)
        sphere_samples = gaussian_samples / norms[:, None]

        # Add isotropic gaussian noise
        noise = mvn.rvs(mean=np.zeros(self.p), cov=self.sigma2, size=n)
        samples = sphere_samples + noise

        return samples


if __name__ == "__main__":
	import matplotlib.pyplot as plt

    n = 100
    p = 2

    plt.figure(figsize=(7, 7))
    sigma2_range = [np.power(10.0, x) for x in np.arange(-2, 2)]
    for ii, sigma2 in enumerate(sigma2_range):
        sfg = StandardFisherGaussian(p=p, sigma2=sigma2)
        sfg_samples = sfg.sample(n)

        plt.subplot(2, 2, ii + 1)
        plt.scatter(sfg_samples[:, 0], sfg_samples[:, 1])
        plt.title("sigma2={}".format(sigma2))

    plt.show()

