'''

Some partially defined functions for the Machine Learning assignment.

You should complete the provided functions and add more functions and classes as necessary.

Write a main function that calls different functions to perform the required tasks.

'''

# Imports
import numpy as np
from sklearn import preprocessing, model_selection, neighbors, tree, svm
from sklearn.naive_bayes import *
import sklearn


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)

    '''
    return [(9493671, 'Daniel', 'Huffer'), (1234568, 'Jenyfer', 'Florentina')]


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def prepare_dataset(dataset_path):
    '''
    Read a comma separated text file where
	- the first field is a ID number
	- the second field is a class label 'B' or 'M'
	- the remaining fields are real-valued

    Return two numpy arrays X and y where
	- X is two dimensional. X[i,:] is the ith example
	- y is one dimensional. y[i] is the class label of X[i,:]
          y[i] should be set to 1 for 'M', and 0 for 'B'

    @param dataset_path: full path of the dataset text file

    @return
	X,y
    '''

    # small function f(x) for the lambda expression inside np.loadtxt for array Y
    def f(x):
        if x == b'M':
            return 1
        else:
            return 0

    X = np.loadtxt('medical_records.data', delimiter=",", usecols=(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, \
                                                                   17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                                                                   30, 31))
    Y = np.loadtxt('medical_records.data', delimiter=",", usecols=(1,), dtype="str", \
                   converters={1: lambda s: f(s)})
    return X, Y


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NB_classifier(X_data, y_data):
    '''
    Build a Naive Bayes classifier based on the training set X_training, y_training.

    @param
	X_data: X_data[i,:] is the ith example
	y_data: y_data[i] is the class label of X_data[i,:]

    @return
	clf : the classifier built in this function
    '''
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X_data, y_data, test_size=0.1, \
                                                                        random_state=42)
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    print('Accuracy on the training subset for NB: {:3f}'.format(clf.score(X_train, y_train)))
    print('Accuracy on the testing subset for NB: {:3f}'.format(clf.score(X_test, y_test)))
    return clf

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_DT_classifier(X_data, y_data):
    '''
    Build a Decision Tree classifier based on the training set X_training, y_training.

    @param
	X_data: X_data[i,:] is the ith example
	y_data: y_data[i] is the class label of X_data[i,:]

    @return
	clf : the classifier built in this function
    '''

    #make a range of tree depth values for testing optimal DT depth
    depth_range = range(1, 10)
    #array to hold the results
    depth_scores = []
    #Shuffle the data with random_state = 42, Put away 15 percent of the data straight away with test_size (15 percent
    #NOT touched until the very END) ,
    #Feed in entire data set into train_test_split
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X_data, y_data, test_size=0.15, \
                                                                        random_state=42)
    for i in depth_range: #loop all the tree values
        dt = tree.DecisionTreeClassifier(max_depth=i, random_state=0) #make the classifier
        scores = model_selection.cross_val_score(dt, X_train, y_train, cv=10, scoring='accuracy')#Use cv=10, 10 fold cross
        #validation of training data to find best parameters
        depth_scores.append(scores.mean())#record the mean average of the 10 fold training for each tree depth limit
    import matplotlib.pyplot as plt
    plt.plot(depth_range, depth_scores)#plot results of training for best parameter
    plt.xlabel('Depth of tree for DT')
    plt.ylabel('Test subset accuracy')
    plt.show()
    best_d = np.argmax(depth_scores)#take the best param
    print("best depth of tree is : ", best_d, "with mean average cross validation accuracy of : ", depth_scores[best_d])
    clf = tree.DecisionTreeClassifier(max_depth=best_d, random_state=0)  #make the classifier with the best param
    clf.fit(X_train, y_train)#fit the data to the classifier
    print("Accuracy of DT classifier on unseen test data: ",clf.score(X_test,y_test))
    return clf



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NN_classifier(X_data, y_data):
    '''
    Build a Nearrest Neighbours classifier based on the training set X_training, y_training.

    @param
	X_data: X_data[i,:] is the ith example
	y_data: y_data[i] is the class label of X_data[i,:]

    @return
	clf : the classifier built in this function
    '''
    # 10-fold cross validation with K=5 for KNN
    # search for an optiomal value of K for KNN
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X_data, y_data, test_size=0.15, \
                                                                        random_state=42)
    k_range = range(1, 31)
    k_scores = []
    for k in k_range:
        knn = neighbors.KNeighborsClassifier(n_neighbors=k) #looping k variable finding best parameter
        scores = model_selection.cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy') #10 fold cross validation
        k_scores.append(scores.mean()) #mean of 10 fold result
    import matplotlib.pyplot as plt
    plt.plot(k_range, k_scores)
    plt.xlabel('Value of k for KNN')
    plt.ylabel('Cross-Validated Accuracy')
    plt.show()
    best_k = np.argmax(k_scores) #take best score param
    print("best k value is : ", best_k, "with mean average cross validation accuracy of : ", k_scores[best_k])
    clf = neighbors.KNeighborsClassifier(n_neighbors=best_k) #use best score param
    clf.fit(X_train, y_train) #fit the training data
    print("Accuracy of KNN classifier on unseen test data: ", clf.score(X_test, y_test)) #test the classifier
    return clf

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_SVM_classifier(X_training, y_training):
    '''
    Build a Support Vector Machine classifier based on the training set X_training, y_training.

    @param
	X_data: X_data[i,:] is the ith example
	y_data: y_data[i] is the class label of X_data[i,:]

    @return
	clf : the classifier built in this function
    '''


    def svc_param_selection(X, y, nfolds):
        '''
        Build a Support Vector Machine classifier based on the training set X_training, y_training.

        @param
        X: X training data
        y: y training data
        nFolds: Number of folds for cross validation

        @return
        gridsearch.best_params_ : the best parameters for the classifier based on grid search results
        '''
        Cs = [0.001, 0.01, 0.1, 1, 10] #grid of values incrementing by power of 10
        gammas = [0.001, 0.01, 0.1, 1]#grid of values incrementing by power of 10
        param_grid = {'C': Cs, 'gamma': gammas}#grid dictionary
        gridsearch = model_selection.GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds) #perform exhaustive search
        gridsearch.fit(X, y)
        return gridsearch.best_params_

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X_training, y_training, test_size=0.15, \
                                                                        random_state=42)
    clf = svm.SVC()
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print("Accuracy of SVM classifier using default params: ", accuracy)

    params = svc_param_selection(X_train, y_train, 10) #select best params function with 10 fold cross validation
    clf = svm.SVC(C=params['C'], gamma=params['gamma'])
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print("Accuracy of SVM classifier using tuned params: ", accuracy)
    return clf


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if __name__ == "__main__":
    # call your functions here
    X_data, Y_data = prepare_dataset("medical_records.data")
    build_NB_classifier(X_data, Y_data)
    build_NN_classifier(X_data, Y_data)
    build_DT_classifier(X_data, Y_data)
    build_SVM_classifier(X_data, Y_data)



