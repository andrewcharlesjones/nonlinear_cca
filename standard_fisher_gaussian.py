import numpy as np
import math
from scipy.special import gamma, iv


class StandardFisherGaussian:

	def __init__(self, p, sigma2):
		self.p = p
		self.sigma2 = sigma2

	# evaluate density
	def pdf(self, x):
		# TODO: write down density function
		# nu is the order of the Modified Bessel Function of the first kind, see the sentence below "else"
		nu = p/2-1
		a = np.sqrt(np.inner(x,x))/sigma2
		
		# when x is at the origin (zero norm), evaluating the denominator of the density will result in zero but the limit exists 
		# so the density function is continuous everywhere
		if a == 0:
			density = (2*math.pi*sigma2)**(-p/2)*math.exp(-1/(2*sigma2))
		else:
			density = 2**(nu)*gamma(p/2)*iv(nu,a)/(a**nu)*(2*math.pi*sigma2)**(-p/2)*math.exp(-(np.inner(x,x)+1)/(2*sigma2))
			
		return density
		# pass

	# sample
	def sample(self, n):

		# Sample from uniform sphere by drawing from isotropic gaussian
		#    and normalizing by L2 norm
		from scipy.stats import multivariate_normal as mvn
		gaussian_samples = mvn.rvs(mean=np.zeros(self.p), cov=1, size=n)
		norms = np.linalg.norm(gaussian_samples, ord=2, axis=1)
		sphere_samples = gaussian_samples / norms[:, None]

		# Add gaussian noise with variance sigma2
		noise = mvn.rvs(mean=np.zeros(self.p), cov=self.sigma2, size=n)
		samples = sphere_samples + noise

		return samples


if __name__ == "__main__":
	import matplotlib.pyplot as plt

	n = 1000
	p = 2

	plt.figure(figsize=(7, 7))
	sigma2_range = [np.power(10.0, x) for x in np.arange(-6, 6)]
	num_subplot_rows = np.ceil(np.sqrt(len(sigma2_range)))
	for ii, sigma2 in enumerate(sigma2_range):
		sfg = StandardFisherGaussian(p=p, sigma2=sigma2)
		sfg_samples = sfg.sample(n)

		plt.subplot(num_subplot_rows, num_subplot_rows, ii + 1)
		plt.scatter(sfg_samples[:, 0], sfg_samples[:, 1])
		plt.title("sigma2={}".format(sigma2))

	plt.show()

