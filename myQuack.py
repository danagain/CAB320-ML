'''

Some partially defined functions for the Machine Learning assignment. 

You should complete the provided functions and add more functions and classes as necessary.

Write a main function that calls different functions to perform the required tasks.

'''

#Imports
import numpy as np
from sklearn import preprocessing, model_selection, neighbors, tree, svm
from sklearn.naive_bayes import *



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)

    '''
    return [ (9493671, 'Daniel', 'Huffer'), (1234568, 'Jenyfer', 'Florentina') ]



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
    ##         "INSERT YOUR CODE HERE"

    # small function f(x) for the lambda expression inside np.loadtxt for array Y
    def f(x):
        if x == b'M':
            return 1
        else:
           return 0
   # Load the two arrays
    X = np.loadtxt('medical_records.data', delimiter=",", usecols = (2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,\
                                                                     17,18,19,20,21,22,23,24,25,26,27,28,29,30,31))
    Y = np.loadtxt('medical_records.data', delimiter = ",", usecols = (1,), dtype="str",\
                   converters = { 1: lambda s: f(s)})
    return X,Y




# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NB_classifier(X_training, y_training):
    '''  
    Build a Naive Bayes classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE"
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X_training,y_training,test_size=0.1,\
                                                                        random_state = 42)
    clf = BernoulliNB()
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print(accuracy)



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_DT_classifier(X_training, y_training):
    '''  
    Build a Decision Tree classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE"
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X_training,y_training,test_size=0.1,\
                                                                        random_state = 42)
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print(accuracy)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NN_classifier(X_training, y_training):
    '''  
    Build a Nearrest Neighbours classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE"


    #10-fold cross validation with K=5 for KNN
    # search for an optiomal value of K for KNN
    k_range = range(1,31)
    k_scores = []
    for k in k_range:
        knn = neighbors.KNeighborsClassifier(n_neighbors=k)
        scores = model_selection.cross_val_score(knn, X_training, y_training, cv=10, scoring='accuracy')
        k_scores.append(scores.mean())
    #print(k_scores)

    import matplotlib.pyplot as plt
    plt.plot(k_range, k_scores)
    plt.xlabel('Value of k for KNN')
    plt.ylabel('Cross-Validated Accuracy')
    plt.show()
    best_k = np.argmax(k_scores)
    print("best k value is : ",best_k, "with accuracy of : ",k_scores[best_k])

    #X_train, X_test, y_train, y_test = model_selection.train_test_split(X_training,y_training,test_size=0.2,\
     #                                                                   random_state = 42)
    #clf = neighbors.KNeighborsClassifier(n_neighbors=5)
    #clf.fit(X_train, y_train)
    #accuracy = clf.score(X_test, y_test)
    #print(accuracy)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_SVM_classifier(X_training, y_training):
    '''  
    Build a Support Vector Machine classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE"
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X_training,y_training,test_size=0.2,\
                                                                        random_state = 42)
    clf = svm.SVC()
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print(accuracy)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if __name__ == "__main__":
    # call your functions here
    X_data, Y_data = prepare_dataset("medical_records.data")
    build_NN_classifier(X_data, Y_data)
    build_NB_classifier(X_data, Y_data)
    build_DT_classifier(X_data, Y_data)
    build_SVM_classifier(X_data, Y_data)



