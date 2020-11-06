import numpy as np
import math
from scipy.special import gamma, iv


class GeneralizedFisherGaussian:

	def __init__(self, d, D, V, c, r, sigma2):
		self.d = d
		self.D = D
		self.V = V
		self.c = c
		self.r = r
		self.sigma2 = sigma2

	# evaluate density
	def pdf(self, x):
		pass

	# sample
	def sample(self, n):

		# Sample from uniform sphere by drawing from isotropic gaussian
		#    and normalizing by L2 norm
		from scipy.stats import multivariate_normal as mvn
		gaussian_samples = mvn.rvs(mean=np.zeros(self.d), cov=1, size=n)

		# If d=1, then need to make this two-dimensional
		if len(gaussian_samples.shape) == 1:
			gaussian_samples = np.expand_dims(gaussian_samples, axis=1)

		# import ipdb; ipdb.set_trace()
		norms = np.linalg.norm(gaussian_samples, ord=2, axis=1)
		z = gaussian_samples / norms[:, None] # these are the uniform sphere samples

		# Project from low-dim to high-dim
		x_nonoise = self.r * self.V @ z.T + np.expand_dims(self.c, axis=1) # Need to expand dims to broadcast across samples

		# Add gaussian noise with variance sigma2
		epsilon = mvn.rvs(mean=np.zeros(self.D), cov=self.sigma2, size=n)
		x = x_nonoise + epsilon.T

		# Make it samples x features
		x = x.T

		return x


if __name__ == "__main__":
	import matplotlib.pyplot as plt

	n = 1000
	D = 2
	d = 1
	r = 1
	c = np.zeros(D)
	sigma2 = 0.1
	V = np.random.normal(size=(D, d))

	# Orthogonalize V
	u, dvals, v = np.linalg.svd(V)
	V = u[:, :d]

	V = np.array(
		[[0],
		[1]])

	# V = np.array(
	# 	[[1, 0],
	# 	[0, 1]])

	# Make sure it's close to orthogonal
	# import ipdb; ipdb.set_trace()
	np.testing.assert_array_almost_equal(V.T @ V, np.eye(d))


	# Make distribution
	gfg = GeneralizedFisherGaussian(d=d, D=D, V=V, c=c, r=r, sigma2=sigma2)
	samples = gfg.sample(n=n)

	# plt.scatter(samples[:, 0], samples[:, 1])
	# plt.show()
	# import ipdb; ipdb.set_trace()

	# Plot range of sigma2 with everything else fixed
	sigma2_range = [np.power(10.0, x) for x in np.arange(-3, 3)]
	r_range = [np.power(10.0, x) for x in np.arange(-3, 3)]

	num_subplot_rows = 2
	num_subplot_cols = len(sigma2_range)

	plt.figure(figsize=(20, 20))
	for ii, sigma2 in enumerate(sigma2_range):
		for jj, r in enumerate(r_range):
			gfg = GeneralizedFisherGaussian(d=d, D=D, V=V, c=c, r=r, sigma2=sigma2)
			gfg_samples = gfg.sample(n)

			plt.subplot(num_subplot_cols, num_subplot_cols, ii * num_subplot_cols + jj + 1)
			plt.scatter(gfg_samples[:, 0], gfg_samples[:, 1])
			plt.title("sigma2={}, r={}".format(sigma2, r))

	plt.savefig("./plots/gfg_samples.png")
	plt.show()


	# Plot range of radius with everything else fixed
	# r_range = [np.power(10.0, x) for x in np.arange(-3, 3)]
	# sigma2 = 1

	# for ii, r in enumerate(r_range):
	# 	gfg = GeneralizedFisherGaussian(d=d, D=D, V=V, c=c, r=r, sigma2=sigma2)
	# 	gfg_samples = gfg.sample(n)

	# 	plt.subplot(num_subplot_rows, num_subplot_cols, 1 * num_subplot_cols + ii + 1)
	# 	plt.scatter(gfg_samples[:, 0], gfg_samples[:, 1])
	# 	plt.title("r={}".format(r))


	# plt.show()


	# plt.figure(figsize=(7, 7))
	# sigma2_range = [np.power(10.0, x) for x in np.arange(-6, 6)]
	# num_subplot_rows = np.ceil(np.sqrt(len(sigma2_range)))
	# for ii, sigma2 in enumerate(sigma2_range):
	# 	sfg = StandardFisherGaussian(p=p, sigma2=sigma2)
	# 	sfg_samples = sfg.sample(n)

	# 	plt.subplot(num_subplot_rows, num_subplot_rows, ii + 1)
	# 	plt.scatter(sfg_samples[:, 0], sfg_samples[:, 1])
	# 	plt.title("sigma2={}".format(sigma2))

	# plt.show()

