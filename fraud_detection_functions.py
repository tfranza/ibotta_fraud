import pandas as pd
import numpy as np
import pymysql
from datetime import datetime, timedelta
#For Outlier Detection and Visualization
from pyod.models.iforest import IForest
from pyod.utils.data import generate_data

from pyod.utils.data import evaluate_print
from pyod.utils.example import visualize
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
plt.style.use('ggplot')
from matplotlib.pyplot import figure


def get_transtable(limit_rows): 
	conn = pymysql.connect(host = 'relational.fit.cvut.cz', port = 3306, user = 'guest', password = 'relational', database = 'financial')
	cur = conn.cursor()
	print('Running SQL Query.....')
	cur.execute("SELECT * from trans limit %s", limit_rows)
	results = cur.fetchall()
	#print(cur.description)
	df = pd.DataFrame(list(results), columns = ['trans_id', 'account_id', 'date', 'type', 'operation', 'amount', 'balance', 'k_symbol', 'bank', 'account'])
	return df


def get_feature_null_counts(df): 
	for i, col in enumerate(df.columns):
		column_number = i
		column_name = col
		null_count = df.shape[0] - df[column_name].notnull().sum()
		dtype = df[column_name].dtype
		print(f'{column_number} -> {column_name} -> {null_count} ({dtype})')

to_replace = {
	'type': {'PRIJEM': 'Credit', 'VYDAJ': 'Withdrawal', 'VYBER': 'Withdrawal in cash'},
	'operation': {'VKLAD': 'Credit in cash', 
		    'PREVOD Z UCTU': 'Collection from another bank', 
		    'PREVOD NA UCET': 'Remittance to another bank',
		    'VYBER': 'Withdrawal in cash',
		    'VYBER KARTOU': 'Credit Card Withdrawal'}, 
    'k_symbol': {
		    'SIPO': 'Household Payment', 
		    'SLUZBY': 'Payment of Statement', 
		    'UVER': 'Loan Payment',
		    'POJISTNE': 'Insurance Payment',
		    'DUCHOD': 'Old-age Pension Payment',
		    'UROK': 'Interest Credited',
		    'SANKC. UROK': 'Sanction Interest'
		}
	}

def map_columns(df, to_replace): 
	df['type'] = df['type'].replace(to_replace['type'])
	df['operation'] = df['operation'].replace(to_replace['operation'])
	df['k_symbol'] = df['k_symbol'].replace(to_replace['k_symbol'])
	print(df['type'].unique())
	print(df['operation'].unique())
	print(df['k_symbol'].unique())
	return df


def add_sum_and_count_of_transactions(df, ndays):
	df.sort_values('date', inplace = True)
	df = df[df['type'] == 'Withdrawal']
	grouped = df.groupby('account_id')
	frames = []
	for group in grouped.groups:
	    frame = grouped.get_group(group)
	    #print(frame)
	    frame['sum_{}_days_withdrawals'.format(ndays)] = frame.rolling(ndays, on = 'date').amount.sum()
	    frame['count_{}_days_withdrawals'.format(ndays)] = frame.rolling(ndays, on = 'date').amount.count()
	    frames.append(frame)
	df_w = pd.concat(frames)


	return df_w


# Followed this blog : https://www.justintodata.com/unsupervised-anomaly-detection-on-bank-transactions-outliers/

def train_isolation_forest(df_w, ndays):
	anomaly_proportion = 0.001

	# train IForest detector
	clf_name = 'Anomaly Detection - Isolation Forest'
	clf = IForest(contamination=anomaly_proportion)

	X = df_w[['count_{}_days_withdrawals'.format(ndays), 'sum_{}_days_withdrawals'.format(ndays)]]
	X = np.nan_to_num(X)
	clf.fit(X)

	# get the prediction labels and outlier scores of the training data
	df_w['y_pred'] = clf.labels_ # binary labels (0: inliers, 1: outliers)
	df_w['y_scores'] = clf.decision_scores_
	return clf, df_w


def make_anomaly_chart(df_w_trained, clf, ndays):
	xx , yy = np.meshgrid(np.linspace(0, 11, 200), np.linspace(0, 180000, 200))

	# decision function calculates the raw anomaly score for every point
	Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])*-1
	Z = Z.reshape(xx.shape)
	threshold = (df_w_trained.loc[df_w_trained['y_pred'] == 1, 'y_scores'].min()*-1)/2 + (df_w_trained.loc[df_w_trained['y_pred'] == 0, 'y_scores'].max()*-1)/2
	
	subplot = plt.subplot(1, 1, 1)

	# fill blue colormap from minimum anomaly score to threshold value
	subplot.contourf(xx, yy, Z, levels = np.linspace(Z.min(), threshold, 10),cmap=plt.cm.Blues_r)

	# draw red contour line where anomaly score is equal to threshold
	a = subplot.contour(xx, yy, Z, levels=[threshold],linewidths=2, colors='red')

	# fill orange contour lines where range of anomaly score is from threshold to maximum anomaly score
	subplot.contourf(xx, yy, Z, levels=[threshold, Z.max()],colors='orange')

	msk = df_w_trained['y_pred'] == 0
	x = df_w_trained.loc[msk, ["count_{}_days_withdrawals".format(ndays), "sum_{}_days_withdrawals".format(ndays)]].values

	# scatter plot of inliers with white dots
	b = subplot.scatter(x[:, 0], x[:, 1], c='white',s=20, edgecolor='k') 
	msk = df_w_trained['y_pred'] == 1
	x = df_w_trained.loc[msk, ["count_{}_days_withdrawals".format(ndays), "sum_{}_days_withdrawals".format(ndays)]].values

	# scatter plot of outliers with black dots
	c = subplot.scatter(x[:, 0], x[:, 1], c='black',s=20, edgecolor='r')
	subplot.axis('tight')

	subplot.legend(
    [a.collections[0], b, c],
    ['learned decision function', 'inliers', 'outliers'],
    prop=matplotlib.font_manager.FontProperties(size=10),
    loc='upper right')

	subplot.set_title('Anomaly Detection with Isolation Forest')
	subplot.set_xlim((0, 11))
	subplot.set_ylim((0, 180000))

	subplot.set_xlabel("{}-days count of withdrawal transactions".format(ndays))
	subplot.set_ylabel("{}-days sum of withdrawal transactions".format(ndays))

	plt.savefig('Anomaly Detection - Isolation Forest')


# Tests
df = map_columns(get_transtable(500000), to_replace)
df_w = add_sum_and_count_of_transactions(df, 5)

clf, df_w_trained = train_isolation_forest(df_w, 5)

#df.to_csv('df.csv')
#df_w.to_csv('df_w.csv')
#df_w_trained.to_csv('df_w_trained.csv')

make_anomaly_chart(df_w_trained, clf, 5)