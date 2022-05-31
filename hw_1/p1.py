# name: Qiming Huang
# NUid: 001525526
# problem_1
# part_A
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from sys import float_info
np.random.seed(7)

# ---------- main -----------
def main():
    N = 10000
    mu = np.array([[-0.5, -0.5, -0.5], [1, 1, 1]])
    sigma = np.array([[[1, -0.5, 0.3], [-0.5, 1, -0.5], [0.3, -0.5, 1]],
                      [[1, 0.3, -0.2], [0.3, 1, 0.3], [-0.2, 0.3, 1]]])
    priors = np.array([0.65, 0.35])

    n = mu.shape[1]  # dim: 3
    C = len(priors)  # class: 2
    L = np.array(range(C))  # set labels 0,1

    # thresholds
    thresholds = np.cumsum(priors) # [0.0, 0.65, 1.0]
    thresholds = np.insert(thresholds, 0, 0)
    X = np.zeros([N, n])
    labels = np.zeros(N)

    u = np.random.rand(N)

    # plot
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(111, projection='3d')
    marker_shapes = '..'
    marker_colors = 'rb'
    # scatter loop
    for i in range(C):
        indices = np.argwhere((thresholds[i] <= u) & (u <= thresholds[i + 1]))[:, 0]
        Nl = len(indices)
        labels[indices] = i * np.ones(Nl)
        X[indices, :] = multivariate_normal.rvs(mu[i], sigma[i], Nl)
        ax1.scatter(X[labels == i, 0], X[labels == i, 1], X[labels == i, 2], marker=marker_shapes[i],
                    c=marker_colors[i], label="True Class {}".format(i))

    plt.legend()
    ax1.set_xlabel(r"$x_1$")
    ax1.set_ylabel(r"$x_2$")
    ax1.set_zlabel(r"$x_3$")
    plt.title("p1.1 Generated Original Data Samples")
    plt.tight_layout()
    plt.show()

    Nl = np.array([sum(labels == i) for i in range(C)])
    print("Number of samples: (l=0) = {}, (l=1) = {}".format(Nl[0], Nl[1]))

    # -----------------------------------
    # ROC
    def estimate_roc(discriminant_score, label):
        Nlabels = np.array((sum(label == 0), sum(label == 1)))
        sorted_score = sorted(discriminant_score)
        # Use tau values that will account for every possible classification split
        taus = ([sorted_score[0] - float_info.epsilon] + sorted_score +
                [sorted_score[-1] + float_info.epsilon])
        decisions = [discriminant_score >= t for t in taus]
        # True Positive
        ind11 = [np.argwhere((d == 1) & (label == 1)) for d in decisions]
        p11 = [len(inds) / Nlabels[1] for inds in ind11]
        # False Positive
        ind10 = [np.argwhere((d == 1) & (label == 0)) for d in decisions]
        p10 = [len(inds) / Nlabels[0] for inds in ind10]
        roc = np.array((p10, p11))  # x, then y
        # error probability
        prob_error = [(p10[w] * priors[0] + (1 - p11[w]) * priors[1]) for w in range(len(p10))]

        return roc, taus, prob_error

    # ----------------- Estimated --------------------
    # Expected Risk Minimization Classifier
    class_conditional_likelihoods = np.array([multivariate_normal.pdf(X, mu[l], sigma[l]) for l in L])

    discriminant_score_erm = np.log(class_conditional_likelihoods[1]) - np.log(class_conditional_likelihoods[0])

    roc_erm, tau, prob_error = estimate_roc(discriminant_score_erm, labels)

    # minimum error and index
    minimum_error = min(prob_error)
    minimum_index = prob_error.index(minimum_error)

    # Experimental Threshold Gamma value
    gamma_approx = np.exp(tau[minimum_index])
    print("\nApproximated Threshold: ", gamma_approx)
    print("\nApproximated Minimum Error: ", minimum_error)

    # Minimum Expected Risk Classification (Likelihood-Ratio Test)
    Lambda = np.ones((C, C)) - np.identity(C)
    # Theoretical
    # Gamma threshold for MAP decision rule
    gamma_theoretical = (Lambda[1, 0] - Lambda[0, 0]) / (Lambda[0, 1] - Lambda[1, 1]) * priors[0] / priors[1] # Same as: gamma_th = priors[0]/priors[1]
    print("\nTheoretical Threshold = ", gamma_theoretical)

    # *****  theoretically optimal threshold *****
    decisions_map = discriminant_score_erm >= np.log(gamma_theoretical)

    # True Negative Probability
    ind_00_map = np.argwhere((decisions_map == 0) & (labels == 0))
    p_00_map = len(ind_00_map) / Nl[0]
    # False Positive Probability
    ind_10_map = np.argwhere((decisions_map == 1) & (labels == 0))
    p_10_map = len(ind_10_map) / Nl[0]
    # False Negative Probability
    ind_01_map = np.argwhere((decisions_map == 0) & (labels == 1))
    p_01_map = len(ind_01_map) / Nl[1]
    # True Positive Probability
    ind_11_map = np.argwhere((decisions_map == 1) & (labels == 1))
    p_11_map = len(ind_11_map) / Nl[1]

    # ****** Theoretical *******
    roc_map = np.array((p_10_map, p_11_map))

    # Probability of error for MAP classifier
    # empirically estimated
    prob_error_theoretical = (p_10_map * priors[0] + (1 - p_11_map) * priors[1])
    print("\nTheoretical Minimum Error: ", prob_error_theoretical)

    # Plot ROC
    fig_roc, ax_roc = plt.subplots(figsize=(10, 10))
    ax_roc.plot(roc_erm[0], roc_erm[1])
    ax_roc.plot(roc_erm[0, minimum_index], roc_erm[1, minimum_index], 'rX', label="Experimental Minimum P(Error)", markersize=14)
    ax_roc.plot(roc_map[0], roc_map[1], 'g.', label="Theoretical Minimum P(Error)", markersize=14)
    ax_roc.legend()
    ax_roc.set_xlabel(r"Probability of false alarm $P(D=1|L=0)$")
    ax_roc.set_ylabel(r"Probability of correct decision $P(D=1|L=1)$")
    plt.title("p1.2 Minimum Expected Risk ROC Curve (ERM)")
    plt.grid(True)
    plt.show()

    # MAP decisions
    fig = plt.figure(figsize=(10, 10))
    ax4 = fig.add_subplot(111, projection='3d')

    ax4.scatter(X[ind_00_map, 0], X[ind_00_map, 1], X[ind_00_map, 2], c='r', marker='x', label="Correct Class 0")
    ax4.scatter(X[ind_10_map, 0], X[ind_10_map, 1], X[ind_10_map, 2], c='k', marker='o', label="Incorrect Class 0")
    ax4.scatter(X[ind_01_map, 0], X[ind_01_map, 1], X[ind_01_map, 2], c='b', marker='.', label="Incorrect Class 1")
    ax4.scatter(X[ind_11_map, 0], X[ind_11_map, 1], X[ind_11_map, 2], c='g', marker='+', label="Correct Class 1")

    plt.legend()
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.title("p1.3 MAP Decisions [black & blue -> incorrect]")
    plt.tight_layout()
    plt.show()

    # problem_1
    # part_B
    sigma_nb = np.array([[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, 1]])

    class_conditional_likelihoods_nb = np.array([multivariate_normal.pdf(X, mu[l], sigma_nb) for l in L])
    discriminant_score_erm_nb = np.log(class_conditional_likelihoods_nb[1]) - np.log(
        class_conditional_likelihoods_nb[0])

    roc_erm_nb, tau_nb, prob_error_nb = estimate_roc(discriminant_score_erm_nb, labels)

    # minimum error & index
    minimum_error_nb = min(prob_error_nb)
    print("\nMinimum Error = ", minimum_error_nb)
    minimum_index_nb = prob_error_nb.index(minimum_error_nb)

    # Experimental Threshold Gamma value
    gamma_approx_nb = np.exp(tau_nb[minimum_index_nb])
    print("\nApproximated Threshold = ", gamma_approx_nb)

    # plot
    # ROC
    fig_roc_nb, ax_roc_nb = plt.subplots(figsize=(10, 10))
    ax_roc_nb.plot(roc_erm_nb[0], roc_erm_nb[1])
    ax_roc_nb.plot(roc_erm_nb[0, minimum_index_nb], roc_erm_nb[1, minimum_index_nb], 'rX',
                   label="Experimental Minimum P(Error)", markersize=14)
    ax_roc_nb.legend()
    ax_roc_nb.set_xlabel(r"Probability of false alarm $P(D=1|L=0)$")
    ax_roc_nb.set_ylabel(r"Probability of correct decision $P(D=1|L=1)$")
    plt.grid(True)
    plt.title("p1.4 Minimum Expected Risk ROC Curve - Naive Bayes")
    plt.show()

    # part_C
    # fisher_LDA
    def perform_lda(X, mu, Sigma, C=2):  # (sample, mean vector, cov_matrix, num of class)
        mu = np.array([mu[i].reshape(-1, 1) for i in range(C)])
        cov = np.array([Sigma[i].T for i in range(C)])

        # Determine between class and within class scatter matrix
        Sb = (mu[0] - mu[1]).dot((mu[0] - mu[1]).T)
        Sw = cov[0] + cov[1]

        lambdas, U = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))
        idx = lambdas.argsort()[::-1]
        # Extract corresponding sorted eigenvectors
        U = U[:, idx]
        # First eigenvector
        w = U[:, 0]
        # Scalar LDA projections in matrix form
        z = X.dot(w)

        return w, z

    # mu projection
    mu0proj = np.array([np.average(X[labels == 0][:, i]) for i in range(n)]).T
    mu1proj = np.array([np.average(X[labels == 1][:, i]) for i in range(n)]).T
    mu_proj = np.array([mu0proj, mu1proj])

    sigma0proj = np.cov(X[labels == 0], rowvar=False)
    sigma1proj = np.cov(X[labels == 1], rowvar=False)
    sigma_proj = np.array([sigma0proj, sigma1proj])

    # Fisher LDA Classifier (using true model parameters)
    w, discriminant_score_lda = perform_lda(X, mu_proj, sigma_proj)

    # Estimate the ROC curve for this LDA classifier
    roc_lda, tau_lda, prob_error_lda = estimate_roc(discriminant_score_lda, labels)

    minimum_error_lda = min(prob_error_lda)
    minimum_index_lda = prob_error_lda.index(minimum_error_lda)

    # Experimental Threshold Gamma value
    gamma_approx_lda = np.exp(tau_lda[minimum_index_lda])
    print("\nApproximated LDA Threshold: ", gamma_approx_lda)
    print("\nApproximated LDA Minimum Error: ", minimum_error_lda)

    fig_roc_lda, ax_roc_lda = plt.subplots(figsize=(10, 10))
    ax_roc_lda.plot(roc_lda[0], roc_lda[1])
    ax_roc_lda.plot(roc_lda[0, minimum_index_lda], roc_lda[1, minimum_index_lda], 'gX',
                    label="Experimental Minimum P(Error)", markersize=14)
    ax_roc_lda.legend()
    ax_roc_lda.set_xlabel(r"Probability of false alarm $P(D=1|L=0)$")
    ax_roc_lda.set_ylabel(r"Probability of correct decision $P(D=1|L=1)$")
    plt.title("p1.5 Minimum Expected Risk ROC Curve - Fisher LDA")
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()