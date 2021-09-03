import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap

import matplotlib.pyplot as plt
import seaborn as sns

import scripts.Params

class FeatureEngineer():
    '''
    Provides methods to extract features from the previously cleaned data. Gets exploited during the 
    training step as well as to preprocess new incoming data. 

    '''

    def __init__(self, df: pd.DataFrame):
        '''
        Constructor. Calls the superclass constructor. Creates a copy of the dataframe which to extract 
        the features from, and that is passed as input.
        
        :param df: dataframe to extract features from.
        '''
        super(FeatureEngineer, self).__init__()
        self.df = df.copy(deep=True)
        self.engineered_data = []


    def feature_engineer_data(self, seq_lens = []):
        '''
        Calls the feature engineering operations made available within the class in a sequential way. This 
        specific sequence is currently considered as default for this step.

        Applies pearson correlation feature selection, normalizes the base dataframe, extract transaction 
        sequences as dataframes, normalizes them. Then extracts umap visualization for each dataframe and 
        saves it on the file system.
    
        :param seq_lens: list of lengths for sequences to be generated.
        
        '''
        print()
        print('###########################################')
        print(' Step 2: Feature engineering data')
        print()

        self.df = self.apply_pearson_correlation()

        self.df = self.df.drop(columns=['account_id', 'date'])
        one_length_sequence = self.apply_z_score_normalization(self.df)
        self.df = pd.DataFrame(one_length_sequence, index=self.df.index, columns=self.df.columns)
        
        trans_seqs = [one_length_sequence]
        for seq_len in seq_lens:
            sequence = self.get_transaction_sequences(seq_len)
            sequence = self.apply_z_score_normalization(sequence)
            trans_seq.append(sequence)
        seq_lens = [1] + seq_lens
        
        for seq_len, trans_seq in zip(seq_lens, trans_seqs):
            self.get_umap_visualization (
                df = trans_seq,
                seq_len = seq_len,
                slice_size = 100000
            )
        
    
    #################################################################################################
    ## FEATURE ENGINEERING METHODS

    def apply_pearson_correlation(self, threshold: float = 0.9):
        '''
        Applies feature selection over the cleaned dataframe using the pearson correlation coefficient.
        The unselected features are removed.

        :param threshold: features correlated higher than this threshold will be removed (one of them).

        '''
        print('> Applying pearson correlation coefficient...')

        df_small = self.df.drop(columns=['account_id', 'date'])
        cor = df_small.corr(method='pearson')

        keep_columns = list(range(len(df_small.columns)))
        drop_columns = []
        for i in keep_columns[:-1]:
            for j in keep_columns[i+1:]:
                if np.abs(cor.iloc[i,j]) >= threshold:
                    if j in keep_columns:
                        drop_columns.append(j)
        
        return self.df.drop(columns=df_small.columns[drop_columns])


    def get_transaction_sequences(self, seq_len: int = 2):
        '''
        Retrieves the transaction sequences saved in the file system. If these are not found, then it 
        generates them.

        :param seq_len: number of transactions concatenated in a single sequence.
        
        :returns: the transaction sequences

        '''
        try:
            with open('crafted/sequences_{}trans.pickle'.format(seq_len), 'rb') as handle:
                return pickle.load(handle)
            print('> Loading transaction sequences...')
        except:
            print('> Generating transaction sequences...')
            return build_transaction_sequences(seq_len)


    def build_transaction_sequences(self, seq_len: int = 2):        
        '''
        Builds the transaction sequences from scratch. Considers the base dataframe and extracts windows
        of consequent transactions with length slice_size. Each window belongs to a single account_id.
        Account ids and dates get removed after this operation.

        :param seq_len: number of transactions concatenated in a single sequence.

        :returns: the built sequences of transactions

        '''

        # changes column names as '<n_transaction>_<col_name>' [example: '2_amount' => 2nd transaction of the sequence, field amount]
        cols = np.concatenate([self.df.columns.map(lambda x: str(i)+'_'+x).values for i in range(slice_size)])

        slice_sized_values = [] 
        for account_id in self.df['account_id'].unique():
            df_account = self.df[self.df['account_id']==account_id]
            
            # extracts sequences of length slice_size from dataframe [example: (239, 25) => (238, 2, 25)]
            slices = np.array([df_account.iloc[i:i+slice_size,:].values for i in range(len(df_account)-slice_size+1)])

            # linearizes transactions from the same sequence into single lines [example: (238, 2, 25) => (238, 50)]
            slices = slices.reshape(len(df_account)-slice_size+1, len(cols))

            # stores all sequences of transactions from the single account into the new future dataframe
            slice_sized_values.append(slices)
        
        # concatenates all dataframes from the list of sequences 
        sequences = pd.DataFrame(
            data = np.concatenate(slice_sized_values, axis=0),
            columns = cols
        )
        
        # drops date and account_id columns, useful for the creation of the sequences
        sequences = sequences.drop(columns=[
            col 
            for col in sequences.columns 
            if (col.endswith('date') or col.endswith('account_id'))
        ])
        
        with open('./data/data_analysis/crafted/sequences_{}trans.pickle'.format(slice_size), 'wb') as handle:
            pickle.dump(sequences, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        self.engineered_data.append(zip(seq_len, sequences))
        return sequences

    def apply_z_score_normalization(self, df):
        '''
        Applies feature scaling using the Z score normalization on the dataframe: subtracts the mean, 
        divides with standard deviation.

        :param df: dataframe to be normalized.  

        '''
        return StandardScaler().fit_transform(df)

    def get_umap_visualization(self, df, seq_len: int, slice_size: int = -1):
        '''
        Applies Uniform Manifolf Approximation and Project (UMAP) dimensionality reduction over the dataframe
        passed as input. 

        :param df: dataframe which to get the visualization from.
        :param seq_len: number of transactions concatenated in a single sequence.
        :param slice_size: amount of transactions to slice the original dataframe with. 

        '''
        print('> Generating umap downsampled data for visualization...')
        
        if slice_size != -1:
            df = df[:slice_size,:]

        # generating umap representations and saving them
        umap_trans_df = umap.UMAP().fit_transform(df)
        with open('./data/data_analysis/crafted/umap_{}trans.pickle'.format(seq_len), 'wb') as handle:
            pickle.dump(umap_trans_df, handle, protocol=pickle.HIGHEST_PROTOCOL)        

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
        filepath = './data/data_analysis/crafted/'
        filename = f'umap_{seq_len}trans_plot.png'
        plt.savefig(filepath + filename)
        
    def save_engineered_data(self, filepath: str ='./data/data_analysis/crafted/', filename: str ='df_engineered'):
        '''
        Encodes the engineered dataframe as a bytes object and saves it in the file system according to the given 
        filename and path.

        :param filepath: location path of the object to be stored.
        :param filename: filename of the object to be stored.

        '''
        self.engineered_data = [(1, self.df)] + self.engineered_data
        try:
            with open(f'{filepath}{filename}.pickle', 'wb') as handle:
                pickle.dump(self.df, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('> Feature engineered dataframe saved successfully!')
        except:
            print('> Issue meanwhile trying to save the engineered dataframe...')

