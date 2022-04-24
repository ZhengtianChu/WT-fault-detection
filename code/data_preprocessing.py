import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest,ExtraTreesClassifier, RandomForestClassifier
from sklearn.decomposition import PCA


def zscore_norm(df):
    return (df - df.mean()) / df.std()

def smote(X, Y):
    smo = SMOTE(random_state=42)
    X_smo, Y_smo = smo.fit_sample(X, Y)
    return X_smo, Y_smo
    
    
def random_undersampling(X, Y):
    undersample = RandomUnderSampler(sampling_strategy='majority')
    X_under, Y_under = undersample.fit_resample(X, Y)
    return X_under, Y_under

def ENN(X,Y):
    enn = EditedNearestNeighbours()
    X_res, y_res = enn.fit_resample(X, Y)
    return X_res, y_res
    
    
def MyPCA(X_train,X_test,n):
    pca = PCA(n_components=n)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    return X_train, X_test

    
def feature_selection(X_train):
    corr = X_train.corr()

    # generate heatmap
    #     f, ax= plt.subplots(figsize = (20, 20))
    #     sns.heatmap(corr,cmap='RdBu', linewidths = 0.05, ax = ax)
    #     ax.set_title('Correlation between features')
    #     f.savefig('feature_heatmap.jpg', dpi=100, bbox_inches='tight')

    highThreshold = 0.95
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
    corr_drop = [column for column in upper.columns if any(upper[column].abs() > highThreshold)]


    X_train = X_train.drop(corr_drop,axis=1)
    
    ''' using rf
    clf = RandomForestClassifier()
    clf.fit(X_train, Y_train)
    
    importance = clf.feature_importances_
    indices = np.argsort(importance)[::-1]
    features = X_train.columns

    plt.subplots(figsize = (10, 10))
    plt.barh(features[indices[0:][::-1]], importance[indices[0:][::-1]])
    plt.xlabel( 'Feature Labels' )
    plt.ylabel( 'Feature Importances' )
    plt.title( 'Comparison of different Feature Importances' )
    plt.show()
    '''
    
    ''' using etc
    # Building the model
    extra_tree_forest = ExtraTreesClassifier(n_estimators = 5 , criterion = 'entropy' , max_features = 'auto', random_state = 100 )

    # Training the model
    extra_tree_forest.fit(X_train, Y_train)

    # Computing the importance of each feature
    feature_importance = extra_tree_forest.feature_importances_

    # Normalizing the individual importances
    feature_importance_normalized = np.std([tree.feature_importances_ for tree in 
                                             extra_tree_forest.estimators_], axis = 0 )

    indices = np.argsort(feature_importance_normalized)

    best_features = list(x_smo.columns.values[indices[0:40]])
    feature_importance_normalized = np.sort(feature_importance_normalized)[0:40]

    plt.subplots(figsize = (10, 10))
    plt.barh(best_features, feature_importance_normalized)
    plt.xlabel( 'Feature Labels' )
    plt.ylabel( 'Feature Importances' )
    plt.title( 'Comparison of different Feature Importances' )
    plt.show()
    '''
    
    return X_train