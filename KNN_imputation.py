import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, explained_variance_score
from sklearn.model_selection import train_test_split, cross_val_score

"""
THIS ALGORITHM USES KNN CLASSIFICATION & REGRESSION 
TO IMPUTE ALL MISSING/NaN VALUES IN A DATASET
(Version 1, many things to clean up & improve upon)
"""

# USING THE IRIS DATA SET WITH RANDOM nan VALUES INSERTED TO TEST THE ALGO
names = ['sepal_length','sepal_width','petal_length','petal_width','class']
data = pd.read_csv('iris_with_nans.csv', names=names)
true_data = pd.read_csv('iris.csv', names=names)

# CODE ANY CATEGORICAL DATA HERE PRIOR TO RUNNING THE IMPUTATION FUNCTION
# Want to automate this eventually
data['class'] = data['class'].replace('Iris-setosa', -1)
data['class'] = data['class'].replace('Iris-versicolor', 0)
data['class'] = data['class'].replace('Iris-virginica', 1)
true_data['class'] = true_data['class'].replace('Iris-setosa', -1)
true_data['class'] = true_data['class'].replace('Iris-versicolor', 0)
true_data['class'] = true_data['class'].replace('Iris-virginica', 1)

""" FUNCTION FOR FINDING THE OPTIMAL K """
def get_optimal_k(X, y, col, TYPE):
    neighbors = list(range(1,33,2))
    cv_scores = []
    
    """ FIND ERROR FOR EACH K """
    for k in neighbors:
        if TYPE == 'classification':
            knn = KNeighborsClassifier(n_neighbors=k)
            scoring = 'accuracy'
        elif TYPE == 'regression':
            knn = KNeighborsRegressor(n_neighbors = k)
            scoring = 'neg_mean_squared_error'
        
        # 10 fold Cross Validation to find accuracy of each k
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        scores = cross_val_score(knn, X_train, y_train, cv=10, scoring=scoring)
        cv_scores.append(scores.mean())
    
    # Choose the k with the minimum mean squared error    
    mse = [1 - x for x in cv_scores]
    optimal_k = neighbors[mse.index(min(mse))]
    return(optimal_k)


""" FUNCTION FOR IMPUTING MISSING VALUES FROM A SINGLE COLUMN/PREDICTOR """
def impute_predictor(imputed_dataset, true_data, predictor, knn, TYPE):
    df = imputed_dataset.dropna()
    
    # Make a dataframe with all samples that have missing values we need to impute for the given predictor
    values_to_impute = imputed_dataset.drop(df.index)
    # We only want to impute values on samples where all other values are known or previously imputed
    values_to_impute = values_to_impute.drop(columns=[predictor]).dropna()
    
    # Perform the imputation
    pred = knn.predict(values_to_impute)
    
    # Subset the truth dataset to find the accuracy of our imputation
    truth_values = true_data.iloc[values_to_impute.index]
    imputation_accuracy = pd.DataFrame({"Imputation":pred, "True Val":truth_values[predictor]})
    print(imputation_accuracy)
    
    if TYPE == 'regression':
        error = 1 - np.mean(pow(imputation_accuracy['Imputation'] - imputation_accuracy['True Val'], 2))
        print(f"Mean Squared Error: {error}\n\n")
    elif TYPE == 'classification':
        error = sum(imputation_accuracy['Imputation'] == imputation_accuracy['True Val']) / imputation_accuracy.shape[0]
        print(f"Classification Accuracy: {error}\n\n")
    
    imputed_dataset[predictor][values_to_impute.index] = pred
    return(imputed_dataset)


""" FUNCTION FOR IMPUTING MISSING VALUES OVER AN ENTIRE DATASET """
def knn_imputation(data, true_data):
   
    # LOOK AT HOW MANY NaN VALUES THERE ARE PER PREDICTOR
    cols = list(data.keys())
    num_nans = [data[col].isna().sum() for col in cols]
    
    # SORT THE PREDICTORS BY THE NUM OF NAN's ASSOCIATE WITH EACH
    num_nans, cols = zip(*sorted(zip(num_nans, cols)))
    
    # INITIALIZE THE DATAFRAME WE WANT TO ADD IMPUTATIONS TO
    imputed_dataset = data
    print("\nVALUES IMPUTED: \n")
    
    """ FIT A KNN MODEL FOR EACH PREDICTOR THAT HAS MISSING VALUES """
    for col,nans in zip(cols,num_nans):
        # CHECK TO SEE IF THERE ARE ANY NaN VALUES FOR THAT PREDICTOR
        if nans == 0:
            continue
        
        # BUILD THE TRAINING DATAFRAMES
        df = data.dropna()
        X = df.drop(columns=col)
        y = df[col]
        
        """ DETERMINE IF TARGET PREDICTOR IS CATEGORICAL or NUMERICAL """
        if np.mean([data[col].dropna()%1]) == 0:
            TYPE = 'classification'
            k = get_optimal_k(X, y, col, TYPE)
            knn = KNeighborsClassifier(n_neighbors = k)
        else:
            TYPE = 'regression'
            k = get_optimal_k(X, y, col, TYPE)
            knn = KNeighborsRegressor(n_neighbors = k)
        
        """ TRAIN THE MODEL """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        knn.fit(X_train,y_train)
        pred = knn.predict(X_test)
        
        if TYPE == 'classification':
            accuracy = accuracy_score(y_test, pred)
        else:
            df = pd.DataFrame({'Actual': y_test, 'Prediction':pred, 'Error':np.abs(y_test - pred)})
            accuracy = 1 - np.mean(pow(df['Error'],2))   # 1 - Mean Squared Error
        print("Predictor: ", col, "\nKNN Accuracy: ", accuracy)
      
    
        """ IMPUTATE THE DATA """
        imputed_dataset = impute_predictor(imputed_dataset, true_data, col, knn, TYPE) 
    return(imputed_dataset)
    
    
# Run the algorithm on the iris dataset
imputation = knn_imputation(data, true_data)
