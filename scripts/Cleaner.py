import pandas as pd
import pickle

import Params

class Cleaner():

    def __init__(self, df):
        super(Cleaner, self).__init__()
        self.df = df.copy(deep=True)

    def clean_fields(self):
        self.clean_date_field()
        self.clean_type_field()
        self.clean_operation_field()
        self.clean_ksymbol_field()
        self.clean_bank_field()
        self.clean_account_field()

    #################################################################################################
    ## CLEANERS

    def clean_date_field(self, field_name='date'):
        # extracting series of potentially interesting fields from the date
        dayofweek = self.df['date'].map(lambda date: date.dayofweek)
        day       = self.df['date'].map(lambda date: date.day)
        week      = self.df['date'].map(lambda date: date.week)
        month     = self.df['date'].map(lambda date: date.month)
        year      = self.df['date'].map(lambda date: date.year)

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
        for account_id in self.df['account_id'].unique():
            account_trans = self.df[self.df['account_id']==account_id]['date']
            shifted_trans = account_trans.shift(1, fill_value = account_trans.iloc[0])
            time_between_last_trans = pd.concat([time_between_last_trans,(account_trans-shifted_trans)])
        time_between_last_trans = time_between_last_trans.map(lambda date: date.days)
        date['date_days_from_last_trans'] = time_between_last_trans

        # adding new fields as columns of the original df
        for column in date.columns:
            self.df.insert(
                df.columns.get_loc(field_name),
                column.lower(),
                date[column]
            )


    def clean_type_field(self, field_name='type'):
        # translating type names to make them better readable
        self.df['type'] = self.df['type'].map(Params.TYPES.value)

        # splitting field into three columns and simplifying with these associations: [credit = 100; withdrawal=010; withdrawal_with_cash=011]
        one_hot = pd.get_dummies(df[field_name], prefix=field_name)
        one_hot.columns = ['type_credit', 'type_withdrawal', 'type_cash']
        one_hot.loc[one_hot['type_cash']==1, 'type_withdrawal'] = 1
        for column in one_hot.columns:
            self.df.insert(
                self.df.columns.get_loc(field_name),
                column.lower(),
                one_hot[column]
            )
        self.df.drop(columns=[field_name])


    def clean_operation_field(self, field_name='operation'):
        # translating operation names to make them better readable
        self.df['operation'] = self.df['operation'].map(Params.OPERATIONS.value)

        # transforming operation values into one-hot vectors
        one_hot = pd.get_dummies(self.df[field_name], prefix=field_name)
        one_hot.columns = ['op_credit_from_bank', 'op_withdrawal_from_card', 'op_credit_in_cash', 'op_withdrawal_to_bank', 'op_withdrawal_in_cash']
        for column in one_hot.columns:
            self.df.insert(
                self.df.columns.get_loc(field_name),
                column.lower(),
                one_hot[column]
            )
        self.df.drop(columns=[field_name])


    def clean_ksymbol_field(self, field_name='k_symbol'):
        # translating ksymbol names to make them better readable
        self.df['k_symbol'] = self.df['k_symbol'].map(Params.KSYMBOLS.value)

        # transforming ksymbol values into one-hot vectors
        one_hot = pd.get_dummies(self.df[field_name], prefix=field_name)
        one_hot.columns = ['k_symbol_'+x for x in ['household', 'statement', 'loan', 'insurance', 'pension', 'credited_interest', 'sanction_interest']]
        for column in one_hot.columns:
            self.df.insert(
                self.df.columns.get_loc(field_name),
                column.lower(),
                one_hot[column]
            )
        df.drop(columns=[field_name])

    def clean_bank_field(self, field_name='bank'):
        self.df.drop(columns=[field_name])		# too few infos about this field

    def clean_account_field(self, field_name='account'):
        self.df.drop(columns=[field_name])		# too few infos about this field

    #################################################################################################
    ## UTILITIES

    def save_cleaned_df(self, path='../data/data_analysis/crafted/', filename='df_cleaned'):
        filepath = '{}{}.pickle'.format(path, filename)
        try:
            with open(filepath, 'wb') as handle:
                pickle.dump(self.df, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('Cleaned dataframe saved successfully!')
        except:
            print('Issue meanwhile trying to save the cleaned dataframe...')

    #################################################################################################
