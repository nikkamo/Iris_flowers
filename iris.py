# Following toturial: https://machinelearningmastery.com/machine-learning-in-python-step-by-step/

# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

##############################################################################################################################################################################
#                                       LOAD DATA SET
##############################################################################################################################################################################

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)


##############################################################################################################################################################################
#                                       SUMMARIZE DATA SET
##############################################################################################################################################################################
# look at the data a few different ways: # 1) dimensions of data set, 2) peek at data itself, 3) statistical symmary of attributes, 4) breakdown of data by class variable

# 1) Dimensions of dataset: quick idea of how many instances (rows) and how many attributes (columns) data contains with the shape property:
print( dataset.shape ) # gives (150,5)

# 2) Peek at the data: see the first 20 rows of the data
print( dataset.head(20) )

# 3) Statistical summary: summary of each attribute, including count, mean, min, max, percentiles
print( dataset.describe())

# 4) Class distribution: look at the number of instances (rows) that belong to each class. Can view this as an absolute count
print( dataset.groupby('class').size() ) # each class has same number of instances (50)


##############################################################################################################################################################################
#                                       DATA VISUALIZATION
##############################################################################################################################################################################
# now have basic idea about the data. Extend that with some visualizations by looking at two types of plots:
# 1) Univariate plots to better understand each attribute, 2) multivariate plots to better understand relationship between attributes.

# 1) Univariate plots: plots of each individual variable.

# Since inputs variables are numeric we can create box and whisker plots of each.
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

# Can also create histogram of each input variavle to get idea of the distribution.
dataset.hist() # looks like perhaps two of input variables have gaussian distribution. useful to note as we can use algorithms that can exploit that assumption.
plt.show()

# 2) Multivariate plots: now look at interaction between variables.

# scatterplots of all pars of attribues. can be helpful o spot structured relationships between input variables.
scatter_matrix(dataset) # note diagonal grouping of some pairs of attributes - suggests high correlation and predictable relationship
plt.show()


##############################################################################################################################################################################
#                                       EVALUATE SOME ALGORITHMS
##############################################################################################################################################################################
# now time to create some models of the data and estimate their accuracy on onseen data. This step covers:
# 1) separate out a validation dataset, 2) set-up the test harness to use 10-fold cross validation, 3) build multiple different models to predict species from flower measurements, 3) select the best model.

# 1) Create a validation dataset
array = dataset.values 
X = array[:,0:4]
y = array[:,4]
# split the dataset into two: 80% for training, and 20% for validation
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.2, random_state=1)

# 2) Test harness: will use 10-fold cross validation to estimate model accuracy.
# splits data set into 10 part, train on 9 and test on 1 and repeat for all combinations of train-test splits
# statified means that each fold or splot of the dataset will aim to have the same distribution of example by class as exist in the whole training dataset.
# we set the random seed via the random_state argument to a fixed number to ensure that each algorithm is evaluated on the same splits of the training dataset.
# we are using the metric 'accuracy' to evaluate models.
# This is a ratio of the number of correctly predicted instances divided by the total number of instances in the dataset multiplied by 100 to give a percentage. We will be using the scoring variable when we run build and evaluate each model next.


# 3) Build models: we don't know which algorithms would be good on this problem or what configurations to use. 
# We get an idea from the plots that some of the classes are partially linearly separable in some dimensions so we are expecting generally good results.
# Let's test 6 different algorithms: logistic regression (LR), lnear discriminant analysis (LDA), K-nearet neighbors (KNN), classifiation and regression trees (CART), gaussian naive bayes (NB), support vector machines (SVM). This is a good mix of simple linear (LR, and LDA) and nonlnear (KNN, CART, NV, SVM) agorithms.

...
# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# In this case, we can see that it looks like Support Vector Machines (SVM) has the largest estimated accuracy score at about 0.98 or 98%.
# We can also create a plot of the model evaluation results and compare the spread and the mean accuracy of each model. There is a population of accuracy measures for each algorithm because each algorithm was evaluated 10 times (via 10 fold-cross validation).
# A useful way to compare the samples of results for each algorithm is to create a box and whisker plot for each distribution and compare the distributions.

# Compare Algorithms
plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison')
plt.show()



##############################################################################################################################################################################
#                                       MAKE PREDICTIONS
##############################################################################################################################################################################
# We must choose an algorithm to use to make predictions.
# The results in the previous section suggest that the SVM was perhaps the most accurate model. We will use this model as our final model.
# Now we want to get an idea of the accuracy of the model on our validation set.
# This will give us an independent final check on the accuracy of the best model. It is valuable to keep a validation set just in case you made a slip during training, such as overfitting to the training set or a data leak. Both of these issues will result in an overly optimistic result.

# 1) Make predictions: We can fit the model on the entire training dataset and make predictions on the validation dataset.
# Make predictions on validation dataset
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# 2) Evaluate predictions: We can evaluate the predictions by comparing them to the expected results in the validation set, then calculate classification accuracy, as well as a confusion matrix and a classification report.
# Evaluate predictions
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
# We can see that the accuracy is 0.966 or about 96% on the hold out dataset.
# The confusion matrix provides an indication of the errors made.
# Finally, the classification report provides a breakdown of each class by precision, recall, f1-score and support showing excellent results (granted the validation dataset was small).
