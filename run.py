import pandas as pd

from scripts.Cleaner import Cleaner
from scripts.FeatureEngineer import FeatureEngineer
from scripts.ModelSelector import ModelSelector

orig_df = pd.read_csv(
	'./data/data_analysis/transactions.csv', 
	index_col=['trans_id'], 
	parse_dates=['date'],
	low_memory=False
)

cleaner = Cleaner(orig_df)
cleaner.clean_fields()
cleaner.save_cleaned_df()

df_cleaned = cleaner.df
engineer = FeatureEngineer(df_cleaned)
engineer.feature_engineer_data()
engineer.save_engineered_data()

engineered_data = engineer.engineered_data
for seq_len, eng_df in engineered_data:
	print(seq_len)
	model_selector = ModelSelector(eng_df, seq_len)
	model_selector.train_model('isolation_forests')
	break

