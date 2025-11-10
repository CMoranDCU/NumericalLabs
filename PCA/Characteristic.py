import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import eig

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

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

