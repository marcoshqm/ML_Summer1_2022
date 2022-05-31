# name: Qiming Huang
# NUid: 001525526
# problem_3_Wine_Quality

import matplotlib.pyplot as plt
import numpy as np
import os

from pandas import read_csv
from scipy.stats import multivariate_normal
from sklearn.metrics import confusion_matrix

script_dir = os.path.abspath(os.path.dirname(__file__))
# print(script_dir)

def erm_classifier(X, valid_classes, labels):
    N = len(labels)
    print("\nNumber of samples: ", N)

    N_cl = np.array([sum(labels == i) for i in valid_classes])
    print("\nNumber of Class Labels: \n", N_cl)

    priors = np.array(N_cl / N)
    print("\nPriors: \n", priors)

    C = len(priors)

    # Estimates of mean vector & covariance matrix
    mu_hat = np.array([np.mean(X[labels == i], axis=0) for i in valid_classes])
    reg = 0.1 * np.identity(X.shape[1])
    print("shape_reg", reg.shape)
    Sigma_hat = np.array([np.cov(X[labels == i].T) + reg for i in valid_classes])  # 7x11x11

    # minimum P(error) classification rule
    class_cond_likelihoods = np.array(
        [multivariate_normal.pdf(X, mu_hat[c], Sigma_hat[c]) for c in range(len(valid_classes))])
    priors_diag = np.diag(priors)
    class_posteriors = priors_diag.dot(class_cond_likelihoods)

    decisions = np.argmax(class_posteriors, axis=0)
    decisions = np.array([i + valid_classes[0] for i in decisions])

    conf_matrix_WQ = confusion_matrix(decisions, labels)
    print("\nConfusion_Matrix_(Wine Quality):\n", conf_matrix_WQ)

    error = len(np.argwhere(decisions != labels))
    print('\nNumber of Mis_classifications: ', error, "\nError Estimate: ", error / N)

    # Confusion matrix for Human Activity Recognition
    conf_mat_HAR = np.zeros((C, C))

    for i in decisions:  # decision
        for j in valid_classes:  # class_label
            indices = np.argwhere((decisions == i) & (labels == j))
            conf_mat_HAR[i, j] = round(len(indices) / N_cl[j], 3)
            plt.plot(X[indices, 10], X[indices, 6], 'g.', markersize=6)

            if i != j:
                plt.plot(X[indices, 10], X[indices, 6], 'r.', markersize=6)

    print("\nConfusion_Matrix_(Human Activity Recognition)\n", conf_mat_HAR)
    plt.title("Correct Classification (Green) & Incorrect Classification (Red)")
    plt.xlabel(r"Label 10")
    plt.ylabel(r"Label 3")
    plt.tight_layout()
    plt.show()

    # Plot for original data and their true labels
    fig = plt.figure(figsize=(10, 10))
    marker_shapes = '.......'
    marker_colors = 'rgbkymc'

    for i in valid_classes:
        plt.plot(X[labels == i, 10], X[labels == i, 6], marker_shapes[i] + marker_colors[i],
                 label="True Class {}".format(i))

    plt.legend()
    plt.title("True Labels for two features")
    plt.xlabel(r"Label 10")
    plt.ylabel(r"Label 3")
    plt.tight_layout()
    plt.show()


# PCA Dimensionality Reduction
def pca_reduction(X):
    mu_hat = np.mean(X, axis=0)
    Sigma_hat = np.cov(X.T)

    C = X - mu_hat

    lambdas, U = np.linalg.eig(Sigma_hat)
    idx = lambdas.argsort()[::-1]   # indices_sort
    U = U[:, idx]
    D = np.diag(lambdas[idx])

    Z = np.real(C.dot(U))

    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.scatter(Z[:, 0], Z[:, 1], Z[:, 2], c='mediumblue', s=10)
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.show()

    X_rank = np.linalg.matrix_rank(X)

    rmse = np.zeros(X_rank)
    sum_eigenvals = np.zeros(X_rank)
    no_components = range(1, X_rank + 1)

    # X rejection
    for m in no_components:
        X_hat = Z[:, :m].dot(U[:, :m].T) + mu_hat
        rmse[m - 1] = np.real(np.sqrt(np.mean((X - X_hat) ** 2)))
        sum_eigenvals[m - 1] = np.real(np.sum(D[:m]))

    # Fraction of variance explained
    fraction_var = sum_eigenvals / np.trace(Sigma_hat)

    # MSE plot_1
    plt.figure(1)
    plt.plot(no_components, rmse, color='mediumblue')
    plt.xlabel("Dimensions of PCA")
    plt.ylabel("R_mse")
    plt.show()

    # plot_2
    plt.figure(2)
    plt.plot(no_components, np.real(sum_eigenvals), color='mediumblue')
    plt.xlabel("Dimensions of PCA")
    plt.ylabel("Sum of Eigenvalues")
    plt.show()

    # plot_3
    print("plt_fraction_start")
    plt.figure(3)
    plt.plot(no_components, fraction_var, color='mediumblue')
    plt.xlabel("Dimensions of PCA")
    plt.ylabel("Fraction of Variance Explained")
    plt.show()
    print("plt_fraction_end")


def run_wine_quality_dataset():
    print("[_Wine Quality Process_]\n")
    wine_data = np.array(read_csv(script_dir + '/winequality-white.csv', sep=';'))
    wine_data[:, -1] = np.array([wine_data[i, -1] - 3 for i in range(wine_data.shape[0])])
    labels = wine_data[:, -1]
    valid_classes = np.array([0, 1, 2, 3, 4, 5, 6])
    print("Valid Classes: ", valid_classes)

    X = np.array(wine_data[:, 0:11])
    erm_classifier(X, valid_classes, labels)
    pca_reduction(X)
    print('\n')


#
if __name__ == '__main__':
    run_wine_quality_dataset()
