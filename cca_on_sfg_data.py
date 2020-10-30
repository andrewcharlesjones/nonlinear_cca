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
    k = 2  # latent dimension
    sigma2 = 1
    Psi1 = 1  # covariance of data1
    Psi2 = 1  # covariance of data2

    reps = 5  # num repeated experiments

    # Vary sigma2 parameter - we expect to see traditional CCA behavior as sigma2 approaches 1/k
    sigma2_range = [np.power(10.0, x) for x in np.arange(-6, 6)]
    corr_list = np.zeros((reps, len(sigma2_range)))
    dist = DistanceMetric.get_metric('euclidean')

    for ii in range(reps):
        for jj, sigma2 in enumerate(sigma2_range):

            # Simulate data from SFG
            sfg = StandardFisherGaussian(p=k, sigma2=sigma2)
            z = sfg.sample(n)  # latent variables

            z = (z - np.mean(z, axis=0)) / np.std(z, axis=0)

            W1 = np.random.normal(size=(k, p1))  # loadings
            W2 = np.random.normal(size=(k, p2))  # loadings

            # Sample two datasets for CCA
            X1 = np.array([mvn.rvs(mean=zi@W1, cov=Psi1) for zi in z])
            X2 = np.array([mvn.rvs(mean=zi@W2, cov=Psi2) for zi in z])

            # Run "classic" CCA
            cca = CCA(n_components=k, max_iter=2000)
            cca.fit(X1, X2)


            # Compute recovery of nearest neighbor relationships
            pairwise_dists_data = dist.pairwise(z)
            pairwise_dists_latent = dist.pairwise(cca.x_scores_)

            nn_idx_data = np.array([np.argmin(pairwise_dists_data[ii, :][pairwise_dists_data[ii, :] != 0]) for ii in range(n)])
            nn_idx_latent = np.array([np.argmin(pairwise_dists_latent[ii, :][pairwise_dists_latent[ii, :] != 0]) for ii in range(n)])

            acc = np.mean(nn_idx_data == nn_idx_latent)
            corr_list[ii, jj] = acc

            # Rotate true and fitted factors (using SVD) to make them more comparable
            # uz, dz, vz = np.linalg.svd(W1.T, full_matrices=False)
            # ucca, dcca, vcca = np.linalg.svd(
            #     cca.x_loadings_, full_matrices=False)

            # true_factors = z @ vz.T @ np.diag(dz).T
            # cca_factors = cca.x_scores_ @ vcca.T @ np.diag(dcca).T

            # # Compute correlation between learned factors and true factors
            # corrs_for_heatmap = np.zeros((k, k))
            # for kii in range(k):
            #     for kjj in range(k):
            #         curr_corr = np.abs(
            #             pearsonr(true_factors[:, kii], cca_factors[:, kjj])[0])
            #         corrs_for_heatmap[kii, kjj] = np.abs(curr_corr)
            # # sns.heatmap(corrs_for_heatmap)
            # # plt.show()

            # # Take the maximum of these correlations as a measure of performance (probably need to change this)
            # curr_corr = np.abs(
            #     pearsonr(true_factors[:, 0], cca_factors[:, 0])[0])
            # corr_list[ii, jj] = np.max(corrs_for_heatmap)
            # print("Sigma2 = {}, Correlation = {}".format(
            #     sigma2, round(np.max(corrs_for_heatmap), 3)))

    # Plot performance across different values of sigma2
    plt.errorbar(sigma2_range, np.mean(corr_list, 0),
                 yerr=np.std(corr_list, 0), ecolor="purple")
    plt.axvline(1/k, linestyle='--', c='red')
    ax = plt.gca()
    ax.set_xscale('log')
    plt.xlabel("sigma2")
    plt.ylabel("Performance")

    plt.text(1/k, ax.get_ylim()[1], '1/k')
    plt.savefig("./plots/sfa_cca.png")
    plt.show()
    import ipdb
    ipdb.set_trace()
