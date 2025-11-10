# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 21:17:34 2025

@author: Mono
"""
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import eig

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

def sort_eigens(eigenvalues, eigenvectors):
    # creates a pandas dataframe out of the eigenvectors
    df_eigen = pd.DataFrame(eigenvectors)

    # adds a column for the eigenvalues
    df_eigen['Eigenvalues'] = eigenvalues

    # sorts the dataframe in place by eigenvalue
    df_eigen.sort_values("Eigenvalues", inplace=True, ascending=False)

    # makes a numpy array out of the sorted eigenvalue column
    sorted_eigenvalues = np.array(df_eigen['Eigenvalues'])
    # makes a numpy array out of the rest of the sorted dataframe
    sorted_eigenvectors = np.array(df_eigen.drop(columns="Eigenvalues"))

    # returns the sorted values
    return sorted_eigenvalues, sorted_eigenvectors

def reorient_data(df,eigenvectors):
    # turns the dataframe into a numpy array to enable matrix multiplication
    numpy_data = np.array(df)

    # mutiplies the data by the eigenvectors to get the data in terms of pca features
    pca_features = np.dot(numpy_data, eigenvectors)

    # turns the new array back into a dataframe for plotting
    pca_df = pd.DataFrame(pca_features)

    return pca_df

def Standardize(Arr):
    Mean = 0; 
    Variance = 0;

    for i in range(len(Arr)):
        Mean += Arr[i];

    Mean /= len(Arr);
    for i in range(len(Arr)):
        Variance += (Arr[i] - Mean) ** 2;
        
    Variance /= len(Arr);
    Standard = [];
    for i in range(len(Arr)):
        Standard.append((Arr[i] - Mean) / np.sqrt(Variance));

    #Calculate New Mean.
    StandardMean = 0;
    for i in range(len(Standard)):
        Mean += Standard[i];

    StandardMean /= len(Standard);

    #Calculate New Variance
    StandardVariance = 0;
    for i in range(len(Arr)):
        StandardVariance += (Standard[i] - StandardMean) ** 2;
        
    StandardVariance /= len(Standard);
    return Standard, StandardMean, StandardVariance;

def Covariance(OneArr, OneMean, TwoArr, TwoMean):
    Covar = 0;
    
    for i in range(len(OneArr)):
        Covar += (OneArr[i] - OneMean) * (TwoArr[i] - TwoMean);
    
    Covar /= len(OneArr);
    return Covar;


# load dataset into Pandas DataFrame
df = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])
features = ['sepal length', 'sepal width', 'petal length', 'petal width']

#Getting Data.
SepalLength, SLMean, SLVariance = Standardize(df['sepal length'].to_numpy());
SepalWidth, SWMean, SWVariance = Standardize(df['sepal width'].to_numpy());

PetalLength, PLMean, PLVariance = Standardize(df['petal length'].to_numpy());
PetalWidth, PWMean, PWVariance = Standardize(df['petal width'].to_numpy());

Arr = [SepalLength, SepalWidth, PetalLength, PetalWidth];
ArrMean = [SLMean, SWMean, PLMean, PWMean];
KMatrix = np.zeros((4, 4));

for i in range(len(Arr)):
    for j in range(len(Arr)):
        KMatrix[i][j] = Covariance(Arr[i], ArrMean[i], Arr[j], ArrMean[j]);

#Get the EigenValues, EigenVectors.        
EigenValues, EigenVectors = eig(KMatrix);

EigenValues, EigenVectors = sort_eigens(EigenValues, EigenVectors);
k = 2  # Number of components to keep
selected_eigenvectors = EigenVectors[:, :k]
transformed_data = reorient_data(df[features], selected_eigenvectors);
transformed_df = pd.concat([df, transformed_data.rename(columns={0: 'PC1', 1: 'PC2'})], axis=1)

plt.scatter(transformed_df['PC1'], transformed_df['PC2'], c=df['target'].astype('category').cat.codes)
plt.title("PCA of Iris Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.savefig('Principal_Components.svg');
plt.show()


