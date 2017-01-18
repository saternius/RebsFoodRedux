import pandas as pd
import numpy as np
from sklearn import preprocessing 
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt



#Step 1 ~ Download the data
dataframe_all = pd.read_csv('small.csv');
num_rows = dataframe_all.shape[0]
names = dataframe_all["Shrt_Desc"]

#Step 2 ~ Clean the data
#Remove rows where some collumns have null values
dataframe_all = dataframe_all.dropna();

#Removing unwanted collumns
valid = range(2,49)+[50,52]
dataframe_all = dataframe_all.ix[:,valid]

#Step 3 ~ Creating Feature Vectors
x = dataframe_all.ix[:,:-1].values
standard_scalar = StandardScaler()
x_std = standard_scalar.fit_transform(x)

#t distribution stochastic neighbor embedding (t-SNE) visualization
tsne = TSNE(n_components=2, random_state=0)
x_test_2d = tsne.fit_transform(x_std)

#scatter plot the sample points among 5 classes
markers = ('s','d','o','^','v')
color_map = {0:'red', 1:'blue', 2:'lightgreen',3:'purple',4:'cyan'}
plt.figure()

print(x_test_2d)
print(np.unique(x_test_2d))
for idx, cl in enumerate(x_test_2d):
	x = cl[0]
	y =  cl[1]
	print(x,y)
	plt.scatter(x=x,y=y)
	plt.annotate(names[idx], (x,y))

plt.show()