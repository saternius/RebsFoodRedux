import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt



#Step 1 ~ Download the data
dataframe_all = pd.read_csv('data.csv');
num_rows = dataframe_all.shape[0]

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


# Step 4: get class labels y and then encode it into number 
y = dataframe_all.ix[:,-1].values
# encode the class label
class_labels = np.unique(y)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Step 5: split the data into training set and test set
test_percentage = 0.1
x_train, x_test, y_train, y_test = train_test_split(x_std, y, test_size = test_percentage, random_state = 0)


#Step 6: Run TSNE
#t distribution stochastic neighbor embedding (t-SNE) visualization
tsne = TSNE(n_components=2, random_state=0)
x_test_2d = tsne.fit_transform(x_test)


print x_test_2d
print y_test

#Step 7 Plot
# scatter plot the sample points among 5 classes
markers=('s', 'd', 'o', '^', 'v')
color_map = {0:'red', 1:'blue', 2:'lightgreen', 3:'purple', 4:'cyan'}
plt.figure()
for idx, cl in enumerate(np.unique(y_test)):
	print (idx,cl)
	plt.scatter(x=x_test_2d[y_test==cl,0], y=x_test_2d[y_test==cl,1], label=cl)
plt.xlabel('X in t-SNE')
plt.ylabel('Y in t-SNE')
plt.legend(loc='upper left')
plt.title('t-SNE visualization of test data')
plt.show()
