# name: Qiming Huang
# NUid: 001525526
# problem_2
# part_A
import matplotlib.pyplot as plt  # General Plotting
import numpy as np
from scipy.stats import multivariate_normal

np.random.seed(7)

N = 10000

def main():
    priors = np.array([0.2, 0.25, 0.25, 0.3])
    mu = np.array([[0, 0], [3, 0], [6, 0], [9, 0]])
    sigma = [[[2, 0], [0, 2]],
             [[4, 0], [0, 4]],
             [[6, 0], [0, 6]],
             [[8, 0], [0, 8]]]

    n = mu.shape[1]
    C = len(priors)
    L = np.array(range(C))
    # print(L) # [0 1 2 3]
    Y = np.array(range(C))
    # print(Y)

    thresholds = np.cumsum(priors)
    thresholds = np.insert(thresholds, 0, 0)

    X = np.zeros([N, n])
    labels = np.zeros(N)
    u = np.random.rand(N)

    # plot
    fig = plt.figure(figsize=(10, 10))
    marker_shapes_1 = '....'
    marker_colors_1 = 'rbgk'

    for i in range(C):
        indices = np.argwhere((thresholds[i] <= u) & (u <= thresholds[i + 1]))[:, 0]
        Nl = len(indices)
        labels[indices] = i * np.ones(Nl)
        X[indices, :] = multivariate_normal.rvs(mu[i], sigma[i], Nl)
        plt.plot(X[labels == i, 0], X[labels == i, 1], marker_shapes_1[i - 1] + marker_colors_1[i - 1],
                 label="True Class {}".format(i))

    # Plot the original data and their true labels
    plt.legend()
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.title("p2.1 Generated Original Data Samples")
    plt.tight_layout()
    plt.show()

    Nl = np.array([sum(labels == i) for i in range(C)])
    print("\nNumber of samples: 0 = {}, 1 = {}, 2 = {}, 3 = {}".format(Nl[0], Nl[1], Nl[2], Nl[3]))

    # Lambda Matrix
    lambda_matrix = np.ones((C, C)) - np.identity(C)
    class_cond_likelihoods = np.array([multivariate_normal.pdf(X, mu[j], sigma[j]) for j in Y])
    class_priors = np.diag(priors)
    class_posteriors = class_priors.dot(class_cond_likelihoods)
    cond_risk = lambda_matrix.dot(class_posteriors)
    decisions = np.argmin(cond_risk, axis=0)

    # plot(decision & true labels)
    fig = plt.figure(figsize=(12, 10))
    marker_shapes_2 = 'ox+*.'
    marker_colors_2 = 'rbkmg'

    # Get sample class counts
    sample_class_counts = np.array([sum(labels == j) for j in Y])

    conf_mat = np.zeros((C, C))
    for i in Y:
        for j in Y:
            ind_ij = np.argwhere((decisions == i) & (labels == j))
            conf_mat[i, j] = round(len(ind_ij) / sample_class_counts[j], 3)  # Average over class sample count
            if i == j:
                # True label = Marker shape; Decision = Marker Color
                marker = marker_shapes_2[j] + marker_colors_2[i]
                plt.plot(X[ind_ij, 0], X[ind_ij, 1], 'g' + marker_shapes_2[j], markersize=6,
                         label="Correct decision {}".format(i))
            else:
                plt.plot(X[ind_ij, 0], X[ind_ij, 1], 'r' + marker_shapes_2[j], markersize=6,
                         label="Incorrect Decision {} in label {}".format(i, j))

    print("\nConfusion matrix(A):",conf_mat)

    prob_error = 1 - np.diag(conf_mat).dot(sample_class_counts / N)
    print("\nMinimum Probability of Error(A):", prob_error)

    plt.legend()
    plt.title("p2.2 Minimum Probability of Error Classified Sampled Data")
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.tight_layout()
    plt.show()

    # part_B
    lambda_matrix_B = np.array([[0, 1, 2, 3],
                                [1, 0, 1, 2],
                                [2, 1, 0, 1],
                                [3, 2, 1, 0]])
    cond_risk_B = lambda_matrix_B.dot(class_posteriors)
    decisions_B = np.argmin(cond_risk_B, axis=0)
    print("\nDecisions_B", decisions_B)

    # plot( decision & true labels)
    fig = plt.figure(figsize=(14, 12))
    marker_shapes_3 = 'xo*+p'
    marker_colors_3 = 'rbkgm'

    sample_class_counts = np.array([sum(labels == j) for j in Y])

    # Confusion matrix
    conf_mat_B = np.zeros((C, C))
    for i in Y:
        for j in Y:
            ind_ij = np.argwhere((decisions_B == i) & (labels == j))
            conf_mat_B[i, j] = round(len(ind_ij) / sample_class_counts[j], 3)

            if i == j:
                marker = marker_shapes_3[j] + marker_colors_3[i]
                plt.plot(X[ind_ij, 0], X[ind_ij, 1], 'g' + marker_shapes_3[j], markersize=6,
                         label="Correct decision {}".format(i))
            else:
                plt.plot(X[ind_ij, 0], X[ind_ij, 1], 'r' + marker_shapes_3[j], markersize=6,
                         label="Incorrect Decision {} in label {}".format(i, j))

    print("\nConfusion matrix(B):\n", conf_mat_B)

    prob_error_B = 1 - np.diag(conf_mat_B).dot(sample_class_counts / N)
    print("\nMinimum Probability of Error(B):", prob_error_B)

    plt.legend()
    plt.title("p2.3 Minimum Probability of Error Classified Sampled Data")
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()