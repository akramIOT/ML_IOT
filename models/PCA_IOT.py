
## This is to do PCA  on a High dimension data with multiple  features  and  reduce the dimension by PCA method.
## Number of  features(Y coordinates of the data) in the Input dataset correspond to the  number of  Dimensions for doing this PCA  analysis. 

# Imports
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
import pandas as pd

## PCA  implementation from scratch by using  numpy
class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # Mean centering
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # covariance, function needs samples as columns
        cov = np.cov(X.T)

        # eigenvalues, eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # -> eigenvector v = [:,i] column vector, transpose for easier calculations
        # sort eigenvectors
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        # store first n eigenvectors
        self.components = eigenvectors[0 : self.n_components]

    def transform(self, X):
        # project data
        X = X - self.mean
        # Return Matrix should be a dot Product matrix which should be  Square matrix as per linear algebra.
        return np.dot(X, self.components.T)

# Driver code for  Testing
if __name__ == "__main__":

    # If you want to play around with Iris dataset to solve the same problem of Dimensionality reduction then you can use the iris dataset below.
    # data = datasets.load_digits()
    #data = datasets.load_iris()
    #X = data.data
    #y = data.target

    ## This below sample  IOT  dataset used is  ~ 3.4 Gb and it can be downloaded from https://data.world/cityofchicago/sustainable-green-infrastructure-monitoring-sensors
    
    data = pd.read_csv("Smart_Green_Infrastructure_Monitoring_Sensors_-_Historical.csv", delimiter=None, header='infer', dtype='unicode', low_memory=False)
    ## High number of Samples/rows  in this IOT  dataset as per  shape  output below.
    print(data.shape)
    X, y = data.shape

    # Project the data onto the 2 primary principal components
    pca = PCA(2)
    pca.fit(X)
    X_projected = pca.transform(X)

    print("Shape of X:", X.shape)
    print("Shape of transformed X:", X_projected.shape)

    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]

    plt.scatter(
        x1, x2, c=y, edgecolor="none", alpha=0.8, cmap=plt.cm.get_cmap("viridis", 3)
    )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar()
    plt.show()
