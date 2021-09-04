import pandas as pd

from scripts.Cleaner import Cleaner
from scripts.FeatureEngineer import FeatureEngineer
from scripts.ModelSelector import ModelSelector

from scripts.Params import Params

orig_df = pd.read_csv(
	f'{Params.DATASET_FILEPATH.value}transactions.csv', 
	index_col = ['trans_id'], 
	parse_dates = ['date'],
	low_memory = False
)

cleaner = Cleaner(orig_df, force_rebuild=True)
cleaner.clean_fields()

df_cleaned = cleaner.df
engineer = FeatureEngineer(df_cleaned, force_rebuild=True)
engineer.feature_engineer_data(seq_lens=[1,2], slice_size = Params.ROW_SLICE_SIZE.value)

engineered_data = engineer.engineered_data
for seq_len, eng_df in engineered_data:
	model_selector = ModelSelector(eng_df, seq_len)
	model_selector.train_model('isolation_forests')
	break

