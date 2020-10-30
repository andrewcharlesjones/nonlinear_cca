import numpy as np
import matplotlib.pyplot as plt
from standard_fisher_gaussian import StandardFisherGaussian
from scipy.stats import multivariate_normal as mvn
from sklearn.cross_decomposition import CCA
from scipy.stats import pearsonr
import seaborn as sns
import pandas as pd
from sklearn.neighbors import DistanceMetric


if __name__ == "__main__":
	n = 1000
	p1 = 30
	p2 = 30
	k = 10  # latent dimension
	Psi1 = 1  # covariance of data1
	Psi2 = 1  # covariance of data2

	reps = 5  # num repeated experiments

	dist = DistanceMetric.get_metric('euclidean')

	# Experiments with two conditions: one including squared features, and another not including squared features
	conditions = ['linear', 'squared']
	k_list = np.arange(1, 21)

	performance_list = np.zeros((len(k_list), reps, len(conditions)))

	for kk, k in enumerate(k_list):
		for ii, condition in enumerate(conditions):
			for jj in range(reps):

				# Generate data using quadratic
				z = np.random.normal(size=(n, k))
				W1_lin = np.random.normal(size=(k, p1))
				W1_quad = np.random.normal(size=(k, p1))

				W2_lin = np.random.normal(size=(k, p2))
				W2_quad = np.random.normal(size=(k, p2))

				# X1 = z @ W1_lin + z**2 @ W1_quad
				# X2 = z @ W2_lin + z**2 @ W2_quad

				# Sample two datasets for CCA
				# import ipdb; ipdb.set_trace()
				# X1 = np.array([mvn.rvs(mean=zi @ W1_lin + zi**2 @ W1_quad, cov=Psi1) for zi in z])
				# X2 = np.array([mvn.rvs(mean=zi @ W2_lin + zi**2 @ W2_quad, cov=Psi2) for zi in z])
				X1 = np.array([mvn.rvs(mean=zi @ W1_lin, cov=Psi1) for zi in z])
				X2 = np.array([mvn.rvs(mean=zi @ W2_lin, cov=Psi2) for zi in z])

				if condition == "squared":
					X1 = np.concatenate([X1, X1**2], axis=1)
					X2 = np.concatenate([X2, X2**2], axis=1)

				# Run CCA
				print("Running")
				cca = CCA(n_components=k, max_iter=2000)
				cca.fit(X1, X2)

				# Compute recovery of nearest neighbor relationships
				pairwise_dists_data = dist.pairwise(z)
				pairwise_dists_latent = dist.pairwise(cca.x_scores_)

				nn_idx_data = np.array([np.argmin(pairwise_dists_data[ii, :][pairwise_dists_data[ii, :] != 0]) for ii in range(n)])
				nn_idx_latent = np.array([np.argmin(pairwise_dists_latent[ii, :][pairwise_dists_latent[ii, :] != 0]) for ii in range(n)])

				acc = np.mean(nn_idx_data == nn_idx_latent)
				performance_list[kk, jj, ii] = acc

	# plot_df = pd.melt(pd.DataFrame(performance_list, columns=conditions))

	results_linear = performance_list[:, :, 0]
	plt.errorbar(k_list, np.mean(results_linear, 1),
				 yerr=np.std(results_linear, 1), label="Linear")

	results_square = performance_list[:, :, 1]
	plt.errorbar(k_list, np.mean(results_square, 1),
				 yerr=np.std(results_square, 1), label="Squared")
	plt.legend()
	# sns.boxplot(data=plot_df, x="variable", y="value")
	plt.ylabel("Performance")
	plt.xlabel("Latent dimension")
	plt.savefig("./plots/square_kernel_cca.png")
	plt.show()
	import ipdb
	ipdb.set_trace()


