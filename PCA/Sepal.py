import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import numpy as np

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

    return Standard, Mean, Variance;

# load dataset into Pandas DataFrame
df = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])
features = ['sepal length', 'sepal width', 'petal length', 'petal width']

SepalLength, SLMean, SLVariance = Standardize(df['sepal length'].to_numpy());
SepalWidth, SWMean, SWVariance = Standardize(df['sepal width'].to_numpy());

PetalLength, PLMean, PLVariance = Standardize(df['petal length'].to_numpy());
PetalWidth, PWMean, PWVariance = Standardize(df['petal width'].to_numpy());

#Sepal Length cv
Fig, SLvsSW = plt.subplots();

SLvsSW.scatter(PetalLength, SepalLength);
SLvsSW.set_xlabel('Petal Length (cm)');
SLvsSW.set_ylabel('Sepal Length (cm)');
plt.savefig('PetalLengthvsSepalLength.svg');
plt.show();