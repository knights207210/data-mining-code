# purchasing feature analysis

import pandas as pd
import numpy as np
from numpy import array
import psycopg2
import math
import traceback
import matplotlib.pyplot as plt
from collections import defaultdict

from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn import cross_validation
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# output path
input_path = "C:/Users/yawen/Google Drive/spatial temporal data analytics/urban computing/PurchasingBehavior/feature_analysis_output/"
output_path = "C:/Users/yawen/Google Drive/spatial temporal data analytics/urban computing/PurchasingBehavior/feature_analysis_output/"

# connect to database and get cursor
try:
    conn = psycopg2.connect(database = 'postgres', user = 'postgres', host = 'localhost', port = '5432', password = '123456')

except psycopg2.Error as e:
    print("I am unable to connect to the database")
    print(e)
    print(e.pgcode)
    print(e.pgerror)
    print(traceback.format_exc())

cur = conn.cursor()

def featureAnalysis():
    """
    classify as repeat buyers or not
    """
    cur.execute("select * from user_profile_monthly_count")
    user_profile_features = pd.DataFrame(cur.fetchall(), columns = [i[0] for i in cur.description])

    cur.execute("select * from merchant_profile_monthly_count")
    merchant_profile_features = pd.DataFrame(cur.fetchall(), columns = [i[0] for i in cur.description])

    cur.execute("select * from user_info")
    user_info = pd.DataFrame(cur.fetchall(), columns = [i[0] for i in cur.description])

    cur.execute("select * from train_data")
    train_data = pd.DataFrame(cur.fetchall(), columns = [i[0] for i in cur.description])

    #train_data_user_features = pd.merge(train_data, user_profile_features, on = 'user_id', how = 'left')    #
    #train_data_merchant_features = pd.merge(train_data, merchant_profile_features, on = 'merchant_id', how = 'left')    #
    #train_data_user_info_features = pd.merge(train_data, user_info, on = 'user_id', how = 'left')
   # train_data_user_merchant_profile_features = pd.merge(pd.merge(train_data_user_features, train_data_merchant_features, on = ['user_id', 'merchant_id']), train_data_user_info_features, on = ['user_id', 'merchant_id'])
 
    train_data_user_features = pd.merge(train_data, user_profile_features, on = 'user_id', how = 'left')    
    train_data_merchant_features = pd.merge(train_data, merchant_profile_features, on = 'merchant_id', how = 'left')    
    train_data_user_info_features = pd.merge(train_data, user_info, on = 'user_id', how = 'left')
    train_data_user_merchant_profile_features = pd.merge(pd.merge(train_data_user_features, train_data_merchant_features, on = ['user_id', 'merchant_id']), train_data_user_info_features, on = ['user_id', 'merchant_id'])


    train_data_user_merchant_profile_features = train_data_user_merchant_profile_features.drop('label_x', 1)
    return train_data_user_merchant_profile_features

    # train_data_user_merchant_profile_features_stat = train_data_user_merchant_profile_features.groupby('label').describe()
    # train_data_user_merchant_profile_features_stat.to_csv(output_path + "train_data_user_merchant_profile_features_stat.csv")

    # check missing value perc
    # print(user_info.isnull().sum())
    # print(user_label[user_label['label'] == 1])

    # plot the histgram for columns
    # user_label.hist(column = 'label', bins = 2, grid = False)
    # plt.xticks(np.arange(-2, 2, 1))
    # plt.show()

def confusion_matrix(predictions, test_y):

    d = defaultdict(dict)
    for pp, yy in zip(predictions, test_y):
        d[yy][pp] = d[yy].get(pp, 0) + 1

    return d

def classification(train_data_user_merchant_profile_features):

    x = train_data_user_merchant_profile_features.filter(regex = 'count')
    # ignore NaN, if any value is NaN
    # train_data_user_merchant_profile_features = train_data_user_merchant_profile_features.dropna(how = 'any')

    # x = train_data_user_merchant_profile_features[['gender', 'age_range']]
    y = train_data_user_merchant_profile_features['label']

    accuracy_list  = []
    train_size = [50000, 100000, 200000]
    test_size = [10000, 20000, 50000]

    for i in range(0, 3):
        train_num = train_size[i]
        test_num = test_size[i]

        x_train = x.iloc[:train_num]
        x_test = x.iloc[-test_num:]
        y_train = y.iloc[:train_num]
        y_test = y.iloc[-test_num:]

        model = GradientBoostingClassifier(n_estimators=100)
        model.fit(x_train, y_train)

        predictions = model.predict(x_test)
        err = 1.0 * sum(predictions != y_test)
        accuracy = 1.0 - err / len(predictions)
        print("current accuracy:", accuracy)
        accuracy_list.append(accuracy)

        confusion = confusion_matrix(predictions, y_test)
        print("\t" + "\t".join(str(x) for x in range(0, 2)))
        print("".join(["-"] * 50))
        for ii in range(0, 2):
            jj = ii
            print("%i:\t" % jj + "\t".join(str(confusion[ii].get(x, 0)) for x in range(0, 2)))
        
        #draw ROC, calculate AUC
        mean_tpr = 0.0  
        mean_fpr = np.linspace(0, 1, 100)  
        all_tpr = [] 
        fpr, tpr, thresholds = roc_curve(y_test,predictions)  
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.2f)' % (roc_auc))  
        #plt.plot(fpr, tpr, color='darkorange',lw=3)
        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC %d' % i)
        plt.legend(loc="lower right")
        plt.show()

    print("accuracy list:", accuracy_list)
    print('Finish!')

if __name__ == "__main__":

    train_data_user_merchant_profile_features = featureAnalysis()
    classification(train_data_user_merchant_profile_features)