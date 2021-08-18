import pandas as pd
import pickle

import eif 
from hdbscan import HDBSCAN
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.pca import PCA

import Params

class ModelSelector():
    '''
    Loads and trains the anomaly detection model according to the parameters given in input.

    '''

    def __init__(self, df: pd.DataFrame):
        '''
        Constructor. Calls the superclass constructor. 
        
        :param df: dataframe to extract features from.
        '''
        super(FeatureEngineer, self).__init__()
        self.df = df.copy(deep=True)


    def train_model(self, model_name):
        '''
        Calls the feature engineering operations made available within the class in a sequential way. This 
        specific sequence is currently considered as default for this step.

        Applies pearson correlation feature selection, normalizes the base dataframe, extract transaction 
        sequences as dataframes, normalizes them. Then extracts umap visualization for each dataframe and 
        saves it on the file system.
    
        :param seq_lens: list of lengths for sequences to be generated.

        '''



    #################################################################################################
    ## SELECTED MODELS

    def train_model(self, model_name, data):
        train_model_method = getattr(self, 'self.train_'+model_name)
        labels = train_model_method(data)
        

    def train_knn(self, data):
        knn_model = KNN(
            contamination = 0.03,
            #n_neighbors = 15,
            method = 'median'
        )
        knn_model.fit(data)

        knn_outliers = np.where(knn_model.labels_ == 1)[0]
        knn_inliers = np.where(knn_model.labels_ == 0)[0]

        return knn_outliers, knn_inliers

    def train_local_outlier_factor(self, data):
        lof_model = LOF(
            contamination = 0.03
        )
        lof_model.fit(data)

        lof_outliers = np.where(lof_model.labels_ == 1)[0]
        lof_inliers = np.where(lof_model.labels_ == 0)[0]

        return lof_outliers, lof_inliers

    def train_pca(self, data):
        pca_model = PCA(
            contamination = 0.1,
            whiten = True,
            random_state = 0
        )
        pca_model.fit(data)

        pca_outliers = np.where(pca_model.labels_ == 1)[0]
        pca_inliers = np.where(pca_model.labels_ == 0)[0]

        return pca_outliers, pca_inliers

    def train_hdbscan(self, data, glosh: bool = False):
        clusterer = HDBSCAN(min_cluster_size=15).fit(data)
        
        if glosh:
            threshold = pd.Series(clusterer.outlier_scores_).quantile(0.966)
            labels = (clusterer.outlier_scores_ > threshold).astype(int)
        labels = (clusterer.labels_ == -1).astype(int)

        outliers = np.where(labels == 1)[0]
        inliers = np.where(labels == 0)[0]

        return outliers, inliers


    def train_isolation_forests(self, data):
        ifs_model = IForest(
            contamination=0.03,
            random_state=0
        )
        ifs_model.fit(data)

        outliers = np.where(ifs_model.labels_ == 1)[0]
        inliers = np.where(ifs_model.labels_ == 0)[0]

        return outliers, inliers

    def train_extended_isolation_forest(self, data):
        eif_model = eif.iForest(
            data, 
            ntrees=200, 
            sample_size=256, 
            ExtensionLevel=1
        )
        eif_outliers_probs = eif_model.compute_paths(X_in=data)
        n_outliers = 3000
        
        eif_outliers = eif_outliers_probs.argsort()[-n_outliers:][::-1]
        eif_inliers = eif_outliers_probs.argsort()[:-n_outliers][::-1]
        
        return eif_outliers, eif_inliers

    def train_autoencoders(self, data):
        ae_model = AutoEncoder(
            hidden_neurons = [17,12,12,17],
            epochs = 15,
            contamination = 0.1,
            random_state = 0
        )
        ae.fit(data)

        ae_outliers = np.where(ae_model.labels_ == 1)[0]
        ae_inliers = np.where(ae_model.labels_ == 0)[0]

        return ae_outliers, ae_inliers

