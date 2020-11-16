# Import libraries for analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn import model_selection
from sklearn.model_selection import learning_curve
from sklearn.pipeline import make_pipeline

class InsuranceClassifier:

    # Read in CSV files into pandas df
    df_train = pd.read_csv("C:\\Users\\Jake Ivanov\\Desktop\\Python\\Insurance_Data\\train.csv")
    df_test = pd.read_csv("C:\\Users\\Jake Ivanov\\Desktop\\Python\\Insurance_Data\\test.csv")

    # Get Data characteristics
    df_train.shape, df_train.head()
    df_test.shape, df_test.head()
    df_train.info(), df_test.info()

    # Function to plot features using horizontal bar graph
    def class_count(x_attr, tt_df):
    
        tt_df[x_attr].value_counts().plot(kind="barh")
        plt.xlabel("Count")
        plt.ylabel("Classes")
        plt.show()
    
        return x_attr, tt_df

    # Call class_count() to count variable attributes from df
    class_count('Gender', df_train)
    class_count('Response', df_train)
    class_count('Vehicle_Age', df_train)

    # Function to preprocess data and change objects to integer to run for our ML models
    def preprocess_data(data, f1, f2, f3):
        """Create dictionary to map int data to string values of column name: Vehicle_Age"""
        dict_1 = {'< 1 Year': 1, '1-2 Year': 2, '> 2 Years': 3}
        """Create dictionary to map int data to string values of column name: Gender"""
        dict_2 = {'Female': 1, 'Male': 0}
        """Create dictionary to map int data to string values of column name: Vehicle_Damage"""
        dict_3 = {'Yes': 1, 'No': 2}

        data[f1] = data[f1].map(dict_1)
        data[f2] = data[f2].map(dict_2)
        data[f3] = data[f3].map(dict_3)

        data

        return data, f1, f2, f3

    # Call function to preprocess data for both training and test datasets
    preprocess_data(df_train, 'Vehicle_Age', 'Gender', 'Vehicle_Damage')
    preprocess_data(df_test, 'Vehicle_Age', 'Gender', 'Vehicle_Damage')

    # Get preprocessed data frame into a new df called train_data for training data (x_train)
    train_data = df_train.drop(['Response', 'id'], axis=1)
    target = df_train.Response  # Get response variable (y_train)
    model_dict = {}  # initialize dict to fit models to training data

    # Function to generate ML models into a dictionary and fit and predict those models.
    # Function returns a dictionary of the classifiers and their predicted values
    def model_selector(model_d, td, response):

        m1 = LogisticRegression(random_state=None, max_iter=1000)
        m2 = Perceptron(random_state=None, eta0=0.1, shuffle=False, fit_intercept=False)
        m3 = AdaBoostClassifier(n_estimators=100, learning_rate=1)
        m4 = RandomForestClassifier(n_estimators=100, bootstrap=True, max_features='sqrt')
        m5 = KNeighborsClassifier(n_neighbors=20)

        model_d = {"clf1": m1, "clf2": m2, "clf3": m3, "clf4": m4, "clf5": m5}

        for k, v in model_d.items():

            # Fit models to the training data and put predictions into dictionary values
            model_d[k] = v.fit(td, response).predict(td)

        return model_d

    # Call a new dictionary from model_selector on the training data without the data being upsampled
    new_dict = model_selector(model_dict, train_data, target)

    def classifier_accuracy(md, td, tg):

        for k, v in md.items():
            # Print accuracy score
            print(f"{k}:Training accuracy score: %.9f" % accuracy_score(tg, md[k]))

    classifier_accuracy(new_dict, train_data, target)

    # Predict on my test data
    # model.predict(test_data)






