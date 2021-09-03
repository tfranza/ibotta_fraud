import pandas as pd
import pickle
import seaborn as sns

from scripts.Params import Params

class Cleaner():
    '''
    Provides methods operating on fields to polish the raw data from the Berka dataset. Gets exploited 
    during the training step as well as to preprocess new incoming data. 

    '''

    def __init__(self, df: pd.DataFrame):
        '''
        Constructor. Calls the superclass constructor. Creates a copy of the dataframe to be cleaned, 
        which is passed as input.
        
        :param df: dataframe to be cleaned.
        '''
        super(Cleaner, self).__init__()
        self.df = df.copy(deep=True)

    def clean_fields(self):
        '''
        Calls the cleaning operations made available with the class in a sequential way. This specific 
        sequence is currently considered as default for this step.

        '''
        print()
        print('###########################################')
        print(' Step 1: Cleaning dataset')
        print()

        df_raw = self.df
        df_with_wrangled_dates = self.clean_date_field(df_raw)
        df_with_wrangled_types = self.clean_type_field(df_with_wrangled_dates)
        df_with_wrangled_ops = self.clean_operation_field(df_with_wrangled_types)
        df_with_wrangled_ksymb = self.clean_ksymbol_field(df_with_wrangled_ops)
        df_with_wrangled_bank = self.clean_bank_field(df_with_wrangled_ksymb)
        df_with_wrangled_acc = self.clean_account_field(df_with_wrangled_bank)
        self.df = df_with_wrangled_acc

    #################################################################################################
    ## CLEANING METHODS

    def clean_date_field(self, df):
        '''
        Extracts relevant information from the 'date' field, among which the basic ones directly inferrable
        (day of week, day, week, month, year) as well as the time gap between a transaction and the previous
        one. Once information is extracted, gets added to the original dataframe. 

        The original date column is kept to make time comparisons between transactions easier - might get used
        into the feature engineering step.
        
        :param df: dataframe to be cleaned
        :returns: the cleaned dataframe over the date field

        '''
        field_name = 'date'
        print(f'> Cleaning {field_name}...')

        # extracting series of potentially interesting fields from the date
        dayofweek = df['date'].map(lambda date: date.dayofweek)
        day       = df['date'].map(lambda date: date.day)
        week      = df['date'].map(lambda date: date.week)
        month     = df['date'].map(lambda date: date.month)
        year      = df['date'].map(lambda date: date.year)

        # renaming headers and concatenating series	
        date = pd.concat(
            objs = [dayofweek, day, week, month, year], 
            axis = 1, 
            keys = ['date_'+x for x in ['dayofweek', 'day', 'week', 'month', 'year']]
        )

        # rescaling year information
        date['date_year'] = date['date_year']-date['date_year'].min()

        # extracting information relative to the time passed from the last transaction for the specific account
        time_between_last_trans = pd.Series(dtype='<M8[ns]')
        for account_id in df['account_id'].unique():
            account_trans = df[df['account_id']==account_id]['date']
            shifted_trans = account_trans.shift(1, fill_value = account_trans.iloc[0])
            time_between_last_trans = pd.concat([time_between_last_trans,(account_trans-shifted_trans)])
        time_between_last_trans = time_between_last_trans.map(lambda date: date.days)
        date['date_days_from_last_trans'] = time_between_last_trans

        # adding new fields as columns of the original df
        cols = df.columns.get_loc(field_name)
        for column in date.columns:
            df.insert(
                cols,
                column.lower(),
                date[column]
            )

        return df 


    def clean_type_field(self, df):
        '''
        Improves the readability of the 'type' field labels (by translating them) and transforms the extracted 
        features into multilabel one-hot columns. Then drops the original 'type' field.

        :param df: dataframe to be cleaned
        :returns: the cleaned dataframe over the type field

        '''
        field_name = 'type'
        print(f'> Cleaning {field_name}...')

        # translating type names to make them better readable
        df['type'] = df['type'].map(Params.TYPES.value)

        # splitting field into three columns and simplifying with these associations: [credit = 100; withdrawal=010; withdrawal_with_cash=011]
        one_hot = pd.get_dummies(df[field_name], prefix=field_name)
        one_hot.columns = ['type_credit', 'type_withdrawal', 'type_cash']
        one_hot.loc[one_hot['type_cash']==1, 'type_withdrawal'] = 1
        for column in one_hot.columns:
            df.insert(
                df.columns.get_loc(field_name),
                column.lower(),
                one_hot[column]
            )
        return df.drop(columns=[field_name])


    def clean_operation_field(self, df):
        '''
        Improves the readability of the 'operation' field labels (by translating them) and transforms the extracted 
        features into one-hot columns. Then drops the original 'operation' field.

        :param df: dataframe to be cleaned
        :returns: the cleaned dataframe over the operation field

        '''
        field_name = 'operation'
        print(f'> Cleaning {field_name}...')

        # translating operation names to make them better readable
        df['operation'] = df['operation'].map(Params.OPERATIONS.value)

        # transforming operation values into one-hot vectors
        one_hot = pd.get_dummies(df[field_name], prefix=field_name)
        one_hot.columns = ['op_credit_from_bank', 'op_withdrawal_from_card', 'op_credit_in_cash', 'op_withdrawal_to_bank', 'op_withdrawal_in_cash']
        for column in one_hot.columns:
            df.insert(
                df.columns.get_loc(field_name),
                column.lower(),
                one_hot[column]
            )
        return df.drop(columns=[field_name])


    def clean_ksymbol_field(self, df):
        '''
        Improves the readability of the 'k_symbol' field labels (by translating them) and transforms the extracted 
        features into one-hot columns. Then drops the original 'k_symbol' field.

        :param df: dataframe to be cleaned
        :returns: the cleaned dataframe over the ksymbol field

        '''
        field_name = 'k_symbol'
        print(f'> Cleaning {field_name}...')

        # translating ksymbol names to make them better readable
        df['k_symbol'] = df['k_symbol'].map(Params.KSYMBOLS.value)

        # transforming ksymbol values into one-hot vectors
        one_hot = pd.get_dummies(df[field_name], prefix=field_name)
        one_hot.columns = ['k_symbol_'+x for x in ['household', 'statement', 'loan', 'insurance', 'pension', 'credited_interest', 'sanction_interest']]
        for column in one_hot.columns:
            df.insert(
                df.columns.get_loc(field_name),
                column.lower(),
                one_hot[column]
            )
        return df.drop(columns=[field_name])


    def clean_bank_field(self, df):
        '''
        Drops the 'bank' field as soon as there are too few infos about this field, as shown in data analysis. 

        :param df: dataframe to be cleaned
        :returns: the cleaned dataframe over the bank field

        '''
        field_name = 'bank'
        print(f'> Cleaning {field_name}...')

        return df.drop(columns=[field_name])		# too few infos about this field


    def clean_account_field(self, df):
        '''
        Drops the 'account' field as soon as there are too few infos about this field, as shown in data analysis.  

        :param df: dataframe to be cleaned
        :returns: the cleaned dataframe over the account field

        '''
        field_name = 'account'
        print(f'> Cleaning {field_name}...')

        return df.drop(columns=[field_name])		# too few infos about this field

    #################################################################################################
    ## UTILITIES

    def save_cleaned_df(self, filepath: str ='./data/data_analysis/crafted/', filename: str ='df_cleaned'):
        '''
        Encodes the cleaned dataframe as a bytes object and saves it in the file system according to the given 
        filename and path.

        :param filepath: location path of the object to be stored.
        :param filename: filename of the object to be stored.

        '''
        try:
            with open(f'{filepath}{filename}.pickle', 'wb') as handle:
                pickle.dump(self.df, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('> Cleaned dataframe saved successfully!')
        except:
            print('> Issue meanwhile trying to save the cleaned dataframe...')

    #################################################################################################
