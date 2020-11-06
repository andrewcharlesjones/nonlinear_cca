import numpy as np
import math
from scipy.special import gamma, iv
from generalized_fisher_gaussian import GeneralizedFisherGaussian


class PSCAGradients:

	def __init__(self, gfg, X):
		self.gfg = gfg

		# Make X features x samples
		self.X = X.T

		self.compute_useful_quantities()

	def softplus(self, x):
		return np.log(1 + np.exp(-np.abs(x))) + np.max(x, 0)

	def compute_useful_quantities(self):
		# Terms that will be useful across gradient calculations
		self.X_centered = self.X - np.expand_dims(gfg.c, axis=1)
		self.X_projected = self.gfg.V.T @ self.X_centered
		self.X_projected_norm = np.linalg.norm(self.X_projected, ord=2, axis=0)
		self.X_projected_norm_std = self.X_projected_norm / self.gfg.sigma2

		# Sort of Gaussian density
		self.exp_term = np.exp(-1/(2 * self.gfg.sigma2) * np.linalg.norm(
			self.X_centered, axis=0, ord=2)**2 + self.gfg.r**2)

		# Orders of bessel functions
		self.nu1 = (self.gfg.d - 1) / 2
		self.nu2 = (self.gfg.d + 1) / 2

	def r_grad(self):

		# Term with Gaussian-ish density
		first_term = self.exp_term / \
			((self.gfg.r * self.X_projected_norm_std)**(self.nu1))

		# Term with modified Bessel functions
		second_term = -self.gfg.r / self.gfg.sigma2 * \
			iv(self.nu1, self.gfg.r * self.X_projected_norm_std) + \
			self.X_projected_norm_std * \
			iv(self.nu2, self.gfg.r * self.X_projected_norm_std)

		# Gradient is product of these two terms
		grad = np.multiply(first_term, second_term)

		# Sum over samples
		grad = np.mean(grad)

		return grad

	def c_grad(self):

		first_term = self.exp_term / \
			(self.gfg.r * self.X_projected_norm_std)**(self.nu1)

		# Term with modified Bessel functions
		second_term = self.X_centered / self.gfg.sigma2 * iv(self.nu1, self.gfg.r * self.X_projected_norm_std) - self.gfg.r * self.gfg.V @ self.X_projected / (self.gfg.sigma2 * self.X_projected_norm) * iv(self.nu2, self.gfg.r * self.X_projected_norm_std)

		# Gradient is product of these two terms
		grad = np.multiply(first_term, second_term)

		# Sum over samples
		grad = np.mean(grad)

		return grad

	def sigma2_grad(self):

		# import ipdb; ipdb.set_trace()
		first_term = self.exp_term / \
			(self.gfg.r * self.X_projected_norm_std)**(self.nu1)

		# Term with modified Bessel functions
		second_term = (np.linalg.norm(self.X_centered, ord=2, axis=0) + self.gfg.r**2) / (2 * self.gfg.sigma2**2) * iv(self.nu1, self.gfg.r *
																													   self.X_projected_norm_std) - self.gfg.r * self.X_projected_norm_std / (self.gfg.sigma2**2) * iv(self.nu2, self.gfg.r * self.X_projected_norm_std)

		# Gradient is product of these two terms
		grad = np.multiply(first_term, second_term)

		# Sum over samples
		grad = np.mean(grad)

		return grad

	def V_grad(self):

		first_term = self.exp_term / \
			(self.gfg.r * self.X_projected_norm_std)**(self.nu1)

		# Term with modified Bessel functions
		# second_term = 2 * self.gfg.r * self.X_centered @ self.X_centered.T @ self.gfg.V / (self.gfg.sigma2 * self.X_projected_norm) * iv((self.gfg.d + 1) / 2, self.gfg.r * self.X_projected_norm_std)

		grads = []
		for ii in range(self.X.shape[1]):
			curr_second_term = 2 * self.gfg.r * np.outer(self.X_centered[:, ii], self.X_centered[:, ii]) @ self.gfg.V / (self.gfg.sigma2 * self.X_projected_norm[ii]) * iv(2, self.gfg.r * self.X_projected_norm_std[ii])
			curr_grad = curr_second_term * first_term[ii]
			grads.append(curr_grad)

		grads = np.array(grads)
		grad = np.mean(grads, axis=0)

		return grad

	def likelihood(self):

		bessel_term = iv(self.nu1, self.gfg.r *
						 self.X_projected_norm_std)
		denominator = self.gfg.r * \
			self.X_projected_norm_std**(self.nu1)

		likelihood = self.exp_term * bessel_term / denominator

		return np.mean(likelihood)

	def gradient_descent(self, n_iter=100, learning_rate=0.1):

		V_init = np.random.normal(size=(self.gfg.D, self.gfg.d))
		c_init = np.random.normal(size=self.gfg.D)
		r_init = np.exp(np.random.normal())
		sigma2_init = np.exp(np.random.normal())

		self.gfg.V = V_init
		self.gfg.c = c_init
		self.gfg.r = r_init
		self.gfg.sigma2 = sigma2_init

		likelihood_trace = []
		r_trace = []
		for iter_num in range(n_iter):

			self.compute_useful_quantities()

			lik = self.likelihood()
			likelihood_trace.append(lik)
			print("Iter: {}, likelihood: {}".format(iter_num, lik))

			self.gfg.r += self.r_grad() * learning_rate
			self.gfg.r = max(self.gfg.r, 1e-6)
			self.compute_useful_quantities()
			self.gfg.c += self.c_grad() * learning_rate
			self.compute_useful_quantities()
			self.gfg.sigma2 += self.sigma2_grad() * learning_rate
			self.compute_useful_quantities()
			self.gfg.sigma2 = max(self.gfg.sigma2, 1e-6)
			self.gfg.V += self.V_grad() * learning_rate
			self.compute_useful_quantities()

			# print(self.gfg.r)
			# print(self.gfg.c)
			# print(self.gfg.sigma2)
			# print(self.gfg.V)

			# print(self.r_grad())
			# print(self.c_grad())
			# print(self.sigma2_grad())
			# print(self.V_grad())

			# import ipdb; ipdb.set_trace()

			

		self.likelihood_trace = likelihood_trace

		# import matplotlib.pyplot as plt
		# plt.plot(r_trace)
		# plt.show()


if __name__ == "__main__":

	n = 1000
	D_true = 2
	d_true = 1
	r_true = 10
	c_true = np.zeros(D_true)
	sigma2_true = 1
	# V = np.random.normal(size=(D, d))

	# Orthogonalize V
	# u, dvals, v = np.linalg.svd(V)
	# V = u[:, :d]

	V_true = np.array(
		[[0],
		[1]])

	# V_true = np.array(
	# 	[[1, 0],
	# 	 [0, 1]])

	# Make sure it's close to orthogonal
	np.testing.assert_array_almost_equal(V_true.T @ V_true, np.eye(d_true))

	# Make distribution
	gfg = GeneralizedFisherGaussian(d=d_true, D=D_true, V=V_true, c=c_true, r=r_true, sigma2=sigma2_true)
	samples = gfg.sample(n=n)

	grad_obj = PSCAGradients(X=samples, gfg=gfg)
	grad_obj.gradient_descent(learning_rate=1, n_iter=1000)

	import matplotlib.pyplot as plt
	plt.plot(grad_obj.likelihood_trace)
	plt.show()
	# import ipdb
	# ipdb.set_trace()

	print(grad_obj.gfg.r)
	print(grad_obj.gfg.c)
	print(grad_obj.gfg.sigma2)
	print(grad_obj.gfg.V)

	samples_learned = grad_obj.gfg.sample(n=n)

	plt.figure(figsize=(14, 6))
	plt.subplot(121)
	plt.scatter(samples[:, 0], samples[:, 1])
	plt.title("Training data")

	plt.subplot(122)
	plt.scatter(samples_learned[:, 0], samples_learned[:, 1])
	plt.title("Sampled data\nfrom fitted model")

	plt.show()
