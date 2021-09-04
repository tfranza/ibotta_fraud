import pandas as pd
import numpy as np
import pickle

import eif 
from hdbscan import HDBSCAN
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.pca import PCA

from scripts.Params import Params


class ModelSelector():
    '''
    Loads and trains the anomaly detection model according to the parameters given in input.

    '''

    def __init__(self, df: pd.DataFrame, seq_len:int = 1):
        '''
        Constructor. Calls the superclass constructor. 
        
        :param df: dataframe to extract features from.
        '''
        super(ModelSelector, self).__init__()
        self.df = df.copy(deep=True)
        self.seq_len = seq_len

        print()
        print('###########################################')
        print(' Step 3: Training model')
        print()


    def train_model(self, model_name):
        '''
        Calls the feature engineering operations made available within the class in a sequential way. This 
        specific sequence is currently considered as default for this step.

        Applies pearson correlation feature selection, normalizes the base dataframe, extract transaction 
        sequences as dataframes, normalizes them. Then extracts umap visualization for each dataframe and 
        saves it on the file system.
    
        :param seq_lens: list of lengths for sequences to be generated.

        '''

        self.model_name = model_name
        self.select_model(model_name, self.df)
        self.save_model_data()

    def select_model(self, model_name, data):
        print(f'> Training {model_name}...')
        train_model_method = getattr(self, f'train_{model_name}')
        model, labels = train_model_method(data)
        self.model = model
        self.outliers, self.inliers = labels


    #################################################################################################
    ## SELECTED MODELS

    def train_knn(self, data):
        knn_model = KNN(
            contamination = Params.CONTAMINATION.value,
            #n_neighbors = 15,
            method = 'median',
            random_state = Params.RANDOM_STATE.value
        )
        knn_model.fit(data)
        knn_outliers = np.where(knn_model.labels_ == 1)[0]
        knn_inliers = np.where(knn_model.labels_ == 0)[0]

        return knn_model, (knn_outliers, knn_inliers)

    def train_local_outlier_factor(self, data):
        lof_model = LOF(
            contamination = Params.CONTAMINATION.value,
            random_state = Params.RANDOM_STATE.value
        )
        lof_model.fit(data)
        lof_outliers = np.where(lof_model.labels_ == 1)[0]
        lof_inliers = np.where(lof_model.labels_ == 0)[0]

        return lof_model, (lof_outliers, lof_inliers)

    def train_pca(self, data):
        pca_model = PCA(
            contamination = 0.1, #Params.CONTAMINATION.value,
            whiten = True,
            random_state = Params.RANDOM_STATE.value
        )
        pca_model.fit(data)
        pca_outliers = np.where(pca_model.labels_ == 1)[0]
        pca_inliers = np.where(pca_model.labels_ == 0)[0]

        return pca_model, (pca_outliers, pca_inliers)

    def train_hdbscan(self, data):
        clusterer = HDBSCAN(min_cluster_size=15).fit(data)
        
        self.glosh = False
        if self.glosh:
            threshold = pd.Series(clusterer.outlier_scores_).quantile(1 - Params.CONTAMINATION.value)
            labels = (clusterer.outlier_scores_ > threshold).astype(int)
        else:
            labels = (clusterer.labels_ == -1).astype(int)
        hdbscan_outliers = np.where(labels == 1)[0]
        hdbscan_inliers = np.where(labels == 0)[0]

        return clusterer, (hdbscan_outliers, hdbscan_inliers)

    def train_isolation_forests(self, data):
        ifs_model = IForest(
            contamination = Params.CONTAMINATION.value,
            random_state = Params.RANDOM_STATE.value
        )
        ifs_model.fit(data)
        ifs_outliers = np.where(ifs_model.labels_ == 1)[0]
        ifs_inliers = np.where(ifs_model.labels_ == 0)[0]

        return ifs_model, (ifs_outliers, ifs_inliers)

    def train_extended_isolation_forest(self, data):
        eif_model = eif.iForest(
            data, 
            ntrees = 200, 
            sample_size = 256, 
            ExtensionLevel = 1
        )
        eif_outliers_probs = eif_model.compute_paths(X_in=data)
        n_outliers = Params.CONTAMINATION.value * Params.ROW_SLICE_SIZE.value
        
        eif_outliers = eif_outliers_probs.argsort()[-n_outliers:][::-1]
        eif_inliers = eif_outliers_probs.argsort()[:-n_outliers][::-1]
        
        return eif_model, (eif_outliers, eif_inliers)

    def train_autoencoders(self, data):
        ae_model = AutoEncoder(
            hidden_neurons = [17,12,12,17],
            epochs = 15,
            contamination = 0.1,  # Params.CONTAMINATION.value,
            random_state = Params.RANDOM_STATE.value
        )
        ae_model.fit(data)
        ae_outliers = np.where(ae_model.labels_ == 1)[0]
        ae_inliers = np.where(ae_model.labels_ == 0)[0]

        return ae_model, (ae_outliers, ae_inliers)


    #################################################################################################
    ## UTILITIES

    def save_model_data(self, filepath: str = Params.OUTPUT_FILEPATH.value, filename: str ='model_data_1trans'):
        '''
        Encodes the data related to the model, including the metadata about the model name, the model itself, 
        the outliers and inliers identified with it
    
        :param filepath: location path of the object to be stored.
        :param filename: filename of the object to be stored.

        '''
        model_data = {
            'model_name': self.model_name,
            'model': self.model,
            'outliers': self.outliers,
            'inliers': self.inliers
        }

        try:
            with open(f'{filepath}{filename}.pickle', 'wb') as handle:
                pickle.dump(model_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('> Model data saved successfully!')
        except:
            print('> Issue meanwhile trying to save the model data...')

    #################################################################################################



