import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap

import matplotlib.pyplot as plt
import seaborn as sns

from scripts.Params import Params


class FeatureEngineer():
    '''
    Provides methods to extract features from the previously cleaned data. Gets exploited during the 
    training step as well as to preprocess new incoming data. 

    '''

    def __init__(self, df: pd.DataFrame, force_rebuild: bool = False):
        '''
        Constructor. Calls the superclass constructor. Creates a copy of the dataframe which to extract 
        the features from, and that is passed as input.
        
        :param df: dataframe to extract features from.
        :param force_rebuild: boolean flag to force rebuilding of artifacts instead of loading them.

        '''
        
        super(FeatureEngineer, self).__init__()
        self.df = df.copy(deep=True)
        self.engineered_data = []
        self.force_rebuild = force_rebuild

        print()
        print('###########################################')
        print(' Step 2: Feature engineering data')
        print()


    def feature_engineer_data(self, seq_lens = [1], slice_size: int = -1):
        '''
        Calls the feature engineering operations made available within the class in a sequential way. This 
        specific sequence is currently considered as default for this step.

        Applies pearson correlation feature selection, normalizes the base dataframe, extract transaction 
        sequences as dataframes, normalizes them. Then extracts umap visualization for each dataframe and 
        saves it on the file system.
    
        :param seq_lens: list of lengths for sequences to be generated.
        :param slice_size: amount of rows to retain from the original dataset. 
        
        '''

        # reducing the amount of rows by slicing the dataset
        if slice_size != -1:
            self.df = self.df.iloc[:slice_size,:]

        # removing correlated columns using pearson indicator
        self.df = self.apply_pearson_correlation(self.df)

        # for each seq_len....
        for seq_len in seq_lens:

            # extract transaction sequences
            sequences = self.get_transaction_sequences(seq_len)
            
            # generate umap representation and visualization
            self.get_umap_representation (df = sequences, seq_len = seq_len)

            # save the generated transaction sequences
            self.save_data(sequences, f'sequences_{seq_len}trans')
            
            # append sequences into engineered data
            self.engineered_data.append((seq_len, sequences))
                
    
    #################################################################################################
    ## FEATURE ENGINEERING METHODS

    def apply_pearson_correlation(self, df: pd.DataFrame, threshold: float = 0.9):
        '''
        Applies feature selection over the cleaned dataframe using the pearson correlation coefficient.
        The unselected features are removed.

        :param threshold: features correlated higher than this threshold will be removed (one of them).

        '''
        print('> Applying pearson correlation coefficient...')

        df_small = df.drop(columns=['account_id', 'date'])
        cor = df_small.corr(method='pearson')

        keep_columns = list(range(len(df_small.columns)))
        drop_columns = []
        for i in keep_columns[:-1]:
            for j in keep_columns[i+1:]:
                if np.abs(cor.iloc[i,j]) >= threshold:
                    if j in keep_columns:
                        drop_columns.append(j)
        
        return df.drop(columns=df_small.columns[drop_columns])



    def get_transaction_sequences(self, seq_len: int = 2):
        '''
        Retrieves the transaction sequences saved in the file system. If these are not found, then it 
        generates them.

        :param seq_len: number of transactions concatenated in a single sequence.
        
        :returns: the transaction sequences

        '''
        try:
            if self.force_rebuild:
                raise Exception()

            print()
            print(f'\n> Loading transaction sequences of length {seq_len}...')
            self.load_data(f'sequences_{seq_len}trans')
        except:
            print(f'\n> Generating transaction sequences of length {seq_len}...')
            return self.build_transaction_sequences(seq_len)

    def get_umap_representation(self, df: pd.DataFrame, seq_len: int):
        '''
        Retrieves the umap representation saved in the file system. If this is not found, then it 
        generates them.

        :param seq_len: number of transactions concatenated in a single row.
        
        :returns: the umap representation

        '''
        try:
            if self.force_rebuild:
                raise Exception()

            print(f'> Loading umap representation for sequences of length {seq_len}...')
            self.load_data(f'umap_{seq_len}trans')
        except:
            print(f'> Generating umap representation for sequences of length {seq_len}...')
            return self.build_umap_representation(df, seq_len)



    def build_transaction_sequences(self, seq_len: int = 2):        
        '''
        Builds the transaction sequences from scratch. Considers the base dataframe and extracts windows
        of consequent transactions with length slice_size. Each window belongs to a single account_id.
        Account ids and dates get removed after this operation.

        :param seq_len: number of transactions concatenated in a single row.

        :returns: the built sequences of transactions

        '''

        if seq_len != 1:
            # changes column names as '<n_transaction>_<col_name>' [example: '2_amount' => 2nd transaction of the sequence, field amount]
            cols = np.concatenate([self.df.columns.map(lambda x: str(i)+'_'+x).values for i in range(seq_len)])

            all_accounts_sequences = [] 
            for account_id in self.df['account_id'].unique():
                df_account = self.df[self.df['account_id']==account_id]
                
                # extracts sequences of length slice_size from dataframe [example: (239, 25) => (238, 2, 25)]
                account_sequences = np.array([df_account.iloc[i:i+seq_len,:].values for i in range(len(df_account)-seq_len+1)])

                # linearizes transactions from the same sequence into single lines [example: (238, 2, 25) => (238, 50)]
                account_sequences = account_sequences.reshape(len(df_account)-seq_len+1, len(cols))

                # stores all sequences of transactions from the single account into the new future dataframe
                all_accounts_sequences.append(account_sequences)
        
            # concatenates all dataframes from the list of sequences 
            sequences = pd.DataFrame(
                data = np.concatenate(all_accounts_sequences, axis=0),
                columns = cols
            )        
        else:
            sequences = self.df

        # drops date and account_id columns, useful for the creation of the sequences
        sequences = sequences.drop(columns=[
            col 
            for col in sequences.columns 
            if (col.endswith('date') or col.endswith('account_id'))
        ])

        return sequences

    def build_umap_representation(self, df: pd.DataFrame, seq_len: int):
        '''
        Applies Uniform Manifolf Approximation and Project (UMAP) dimensionality reduction over the dataframe
        passed as input. 

        :param df: dataframe which to get the visualization from.
        :param seq_len: number of transactions concatenated in a single sequence.
        :param slice_size: amount of transactions to slice the original dataframe with. 

        '''

        scaled_df = StandardScaler().fit_transform(df)
        umap_trans_df = umap.UMAP().fit_transform(scaled_df)

        self.save_data(umap_trans_df, f'umap_{seq_len}trans')

        # generating the umap plot
        plt.figure(figsize=(16,10))
        sns.scatterplot(
            x="umap-2d-one", y="umap-2d-two",
            palette=sns.color_palette("hls", 2),
            data=pd.DataFrame(
                data = umap_trans_df,
                columns=['umap-2d-one', 'umap-2d-two']
            ),
            legend="full",
            alpha=0.3
        )

        # saving the umap plot
        filepath = Params.OUTPUT_FILEPATH.value
        filename = f'umap_{seq_len}trans_plot.png'
        plt.savefig(filepath + filename)

    #################################################################################################
    ## UTILITIES
    
    def save_data(self, data_object, filename: str, filepath: str = Params.OUTPUT_FILEPATH.value):
        '''
        Encodes any dataframe or object related to the engineering process that needs to be stored in the file 
        system according to the given filename and filepath.
    
        :param data_object: object to the stored. 
        :param filename: filename of the object to be stored.
        :param filepath: location path of the object to be stored.

        '''
        with open(f'{filepath}{filename}.pickle', 'wb') as handle:
            pickle.dump(data_object, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'> {filename} saved successfully!')


    def load_data(self, filename: str, filepath: str = Params.OUTPUT_FILEPATH.value):
        '''
        Loads encoded dataframe or object related to the engineering process and which is stored in the file 
        system according to the given filename and filepath.

        :param filename: filename of the object to be stored.
        :param filepath: location path of the object to be stored.
        
        :returns: the stored object. 
        '''
        with open(f'{filepath}{filename}.pickle', 'rb') as handle:
            data_object = pickle.load(handle)
        print(f'> {filename} loaded successfully!')

        return data_object

    #################################################################################################
