import pandas as pd
import pickle

import Params

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


    def feature_engineer_data(self, seq_lens = []):
        '''
        Calls the feature engineering operations made available within the class in a sequential way. This 
        specific sequence is currently considered as default for this step.

        Applies pearson correlation feature selection, normalizes the base dataframe, extract transaction 
        sequences as dataframes, normalizes them. Then extracts umap visualization for each dataframe and 
        saves it on the file system.
    
        :param seq_lens: list of lengths for sequences to be generated.

        '''
        self.apply_pearson_correlation()
        
        trans_seqs = [self.apply_z_score_normalization(self.df)]

        for seq_len in seq_lens:
            sequence = self.get_transaction_sequences(seq_len)
            sequence = self.apply_z_score_normalization(sequence)
            trans_seq.append(sequence)
        seq_lens = [1] + seq_lens

        [self.get_umap_visualization(sequence, slice = 100000) for seq_len in seq_lens]


    #################################################################################################
    ## FEATURE ENGINEERING METHODS

    def apply_pearson_correlation(self, threshold: float = 0.9):
        '''
        Applies feature selection over the cleaned dataframe using the pearson correlation coefficient.
        The unselected features are removed.

        :param threshold: features correlated higher than this threshold will be removed (one of them).

        '''
        df_small = self.df.drop(columns=['account_id', 'date'])
        cor = df_small.corr(method='pearson')

        keep_columns = list(range(len(df_small.columns)))
        drop_columns = []
        for i in keep_columns[:-1]:
            for j in keep_columns[i+1:]:
                if np.abs(cor.iloc[i,j]) >= threshold:
                    if j in keep_columns:
                        drop_columns.append(j)

        self.df = self.df.drop(columns=drop_columns)

        return drop_columns


    def get_transaction_sequences(self, seq_len: int = 2):
        '''
        Retrieves the transaction sequences saved in the file system. If these are not found, then it 
        generates them.

        :param seq_len: number of transactions concatenated in a single sequence.

        '''
        try:
            with open('crafted/sequences_{}trans.pickle'.format(seq_len), 'rb') as handle:
                return pickle.load(handle)
        except:
            return build_transaction_sequences(seq_len)


    def build_transaction_sequences(self, seq_len: int = 2):        
        '''
        Builds the transaction sequences from scratch. Considers the base dataframe and extracts windows
        of consequent transactions with length slice_size. Each window belongs to a single account_id.
        Account ids and dates get removed after this operation.

        :param seq_len: number of transactions concatenated in a single sequence.

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
        
        with open('crafted/sequences_{}trans.pickle'.format(slice_size), 'wb') as handle:
            pickle.dump(sequences, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        return sequences

    def apply_z_score_normalization(df):
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
        if slice != -1:
            df = df[:slice,:]

        # generating umap representations and saving them
        umap_trans_df = umap.UMAP().fit_transform(df)
        with open('crafted/umap_{}trans.pickle'.format(seq_len), 'wb') as handle:
            pickle.dump(sequences, handle, protocol=pickle.HIGHEST_PROTOCOL)        

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
        filepath = '../data/data_analysis/crafted/'
        filename = 'umap_{}trans_plot.png'
        plt.savefig(filepath + filename.format(slice))
        
