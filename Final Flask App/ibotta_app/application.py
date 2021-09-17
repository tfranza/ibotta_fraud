# https://www.youtube.com/watch?v=qNF1HqBvpGE
from flask import Flask, render_template, request, redirect, url_for, make_response
import pandas as pd
import numpy as np
import plotly
import plotly.express as px
import plotly.graph_objects as go
from joblib import load
import pickle5 as pickle 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pyod
import csv


application = Flask(__name__, template_folder = 'templates')

models_path = "./models/"
data_path = "./data/"
output_path = "./static/"
anomaly_index = []
account = 0

@application.route('/', methods = ['GET'])
def home_page():
  return render_template('home_page.html')


@application.route('/risk', methods = ['GET'])
def risk_profile():
  # Prepare withdrawals only dataset

  df_cleaned = collect_data_files()
  print('df_cleaned_collected.')
  df_withdrawal = df_cleaned[df_cleaned['type_withdrawal'] == 1]
  df_withdrawal.reset_index(inplace = True)

  # Read the pickle file that contains dictionary with information related to the trained
  # anomaly detection model
  with open(models_path + "isolation_forests_withdrawal.pickle", "rb") as handle:
    ifs_model = pickle.load(handle)

  # Get outlier indices (this is not trans_id). 
  ifs_outliers = np.where(ifs_model.labels_ == 1)[0]

  # Using the outlier index and adding 1 to anomalies based on the dataset that is was
  # trained on - df_withdrawal
  df_withdrawal['anomaly'] = 0
  df_withdrawal.loc[ifs_outliers, ['anomaly']] = 1

  # Calculating risk profiles 
  df_risk = df_withdrawal.groupby('account_id')['anomaly'].mean().reset_index(name = 'Risk')
  df_risk.sort_values(by = 'Risk', ascending = False, inplace = True)  
  df_risk.set_index('account_id', inplace = True)

  # Droping account with no anomalies
  df_risk = df_risk[df_risk.Risk != 0]

  return render_template('risk_profile.html', tables=[df_risk.to_html(classes='data', header="true")])


@application.route('/investigate/<account_id>', methods = ['GET'])
def investigate(account_id):
    # Separate withdrawals

    df_cleaned = collect_data_files()
    df_withdrawal = df_cleaned[df_cleaned['type_withdrawal'] == 1]
    
    # Get the transaction_ids of outliers
    df_withdrawal.reset_index(inplace = True)

    with open(models_path + "isolation_forests_withdrawal.pickle", "rb") as handle:
      ifs_model = pickle.load(handle)

  # Get outlier indices (this is not trans_id). 
    ifs_outliers = np.where(ifs_model.labels_ == 1)[0]
    outlier_trans_id = df_withdrawal['trans_id'].loc[ifs_outliers]

    # Create a new anomaly column and assign 1 to transaction_ids with outliers
    df_cleaned['anomaly'] = 0
    df_cleaned.loc[outlier_trans_id, 'anomaly'] = 1

    df_cleaned_acc = df_cleaned[df_cleaned['account_id'] == int(account_id)]

    make_balance_box(account_id, output_path, df_cleaned_acc)
    make_amount_box(account_id, output_path, df_cleaned_acc)
    make_balance_line(account_id, output_path, df_cleaned_acc)
    #make_side_by_side_balance_box(account_id, output_path, df_cleaned_acc)
    #make_side_by_side_amount_box(account_id, output_path, df_cleaned_acc)
  
    return render_template('investigate.html', 
                          balance = f'/static/balance_box_{account_id}.png',
                         amount = f'/static/amount_box_{account_id}.png', 
                         bal_line = f'/static/balance_line_{account_id}.png')

@application.route('/change', methods = ['POST', 'GET'])
def labels():
  if request.method == 'POST': 
    transaction_id = request.form['transaction_id']
    label = request.form['label']

    fieldnames = ['transaction_id', 'label']
    print("Saving the labelList .......")
    with open('labelList.csv', mode = 'a') as inFile:
      # DictWriter will help you write the file easily by treating the
      # csv as a python's class and will allow you to work with
      # dictionaries instead of having to add the csv manually.
      writer = csv.DictWriter(inFile, fieldnames=fieldnames)
      print("Writing Successfull")

      # writerow() will write a row in your csv file
      
      writer.writerow({'transaction_id': transaction_id, 'label': label})

      # And you return a text or a template, but if you don't return anything
      # this code will never work.
    return "Thank you for the input!"
  else: 
    return render_template('label.html')

# Download the data for the input account ID
# -------------------------------------------------
@application.route("/data/<account_id>", methods = ['GET'])
def getCSV(account_id):
    df_cleaned = collect_data_files()
    df_withdrawal = df_cleaned[df_cleaned['type_withdrawal'] == 1]
    
    # Get the transaction_ids of outliers
    df_withdrawal.reset_index(inplace = True)

    with open(models_path + "isolation_forests_withdrawal.pickle", "rb") as handle:
      ifs_model = pickle.load(handle)

  # Get outlier indices (this is not trans_id). 
    ifs_outliers = np.where(ifs_model.labels_ == 1)[0]
    outlier_trans_id = df_withdrawal['trans_id'].loc[ifs_outliers]

    # Create a new anomaly column and assign 1 to transaction_ids with outliers
    df_cleaned['anomaly'] = 0
    df_cleaned.loc[outlier_trans_id, 'anomaly'] = 1

    df_cleaned_acc = df_cleaned[df_cleaned['account_id'] == int(account_id)]
    
    resp = make_response(df_cleaned_acc.to_csv())
    resp.headers["Content-Disposition"] = "attachment; filename={}_data.csv".format(account_id)
    resp.headers["Content-Type"] = "text/csv"
    return resp
    



#------------------------------------------------------------------------------------
# Collect raw data
def collect_data_files(): 
  with open(data_path + 'df_cleaned.pickle', 'rb') as handle:
      df_cleaned = pickle.load(handle)
  # # Collect feature engineered data
  # with open(data_path + 'sequences_1trans.pickle', 'rb') as handle:
  #     scaled_df = pickle.load(handle)
  # # UMAP for visualization of clusters and anomalies
  # with open(data_path + 'umap_1trans.pickle', 'rb') as handle:
  #     umap_1trans_df = pickle.load(handle)

  return df_cleaned #,scaled_df, umap_1trans_df
#------------------------------------------------------------------------------------


# HOME PAGE
#------------------------------------------------------------------------------------
def make_umap_chart(models_path, output_path, umap_1trans_df):
  # Plot all data points using UMAP (including anomalies from the extended isolation forest model)
  isolation_forests = pickle.load(open(os.path.join(models_path + 'Isolation_forest.pkl'), 'rb'))

  isolation_forests_outliers = np.where(isolation_forests.labels_ == 1)[0]
  isolation_forests_inliers = np.where(isolation_forests.labels_ == 0)[0]

  plt.figure(figsize=(12,8))
  plt.scatter(*umap_1trans_df.T, s=15, linewidth=0, c='gray', alpha=0.25)
  plt.scatter(*umap_1trans_df[isolation_forests_outliers].T, s=15, linewidth=0, c='red', alpha=0.5)
  plt.savefig(output_path + 'anomalies_isolation_forest.png')

  # Return the indices of outliers. 

  #return df[df.index.isin(isolation_forests_outliers)], anomaly_index
  return isolation_forests_outliers

#------------------------------------------------------------------------------------


# Check to see if the user input account id is part of the anomaly detected transactions list. If yes, plot a time series 
# bar chart of 'amount' against the 'date' column
#------------------------------------------------------------------------------------
def make_time_series_chart(account, output_path, df, anomaly_index):
  df.reset_index(inplace = True) 
  df['anomaly'] = -1
  print("printing the original dataframe passed in the function.....")
  print(df.head())
  print(df.shape)
  print("Printing anomaly indices passes to in the function......")
  print(anomaly_index)
  df.loc[anomaly_index, ['anomaly']] = 1
  
  sub_df = df[df['account_id'] == account] 
  print("printing the sub df created using the input account ID.........")
  print(sub_df.head())
  print(sub_df.shape)
  
  fig = go.Figure()
  fig.add_trace(go.Scatter(
    x = sub_df[sub_df['anomaly'] == 1]['date'], 
    y = sub_df[sub_df['anomaly'] == 1]['amount'],
    mode = 'markers',marker = dict(color = 'red')
    ))
  fig.add_trace(go.Scatter(
    x = sub_df[sub_df['anomaly'] == -1]['date'], 
    y = sub_df[sub_df['anomaly'] == -1]['amount'],
    mode = 'markers',marker = dict(color = 'blue')
    ))
  
  fig.write_image(output_path + 'anomalies_for_account_{}.png'.format(int(account)))


def make_balance_box(account_id, output_path, df_cleaned_acc):

  fig1 = px.box(df_cleaned_acc, x = 'anomaly', y = 'balance', points = 'all')
  fig = go.Figure(fig1.data)
  fig.layout.update(
    title = 'Balance (in $) distribution between inliers and outliers',
  xaxis_title = 'Inliers (0) vs Outliers', 
  yaxis_title = 'Balance in $')
  fig.write_image(output_path + f'balance_box_{account_id}.png')


def make_amount_box(account_id, output_path, df_cleaned_acc):
  print(df_cleaned_acc.head(10))
  fig1 = px.box(df_cleaned_acc, x = 'anomaly', y = 'amount', points = 'all')
  fig = go.Figure(fig1.data)
  fig.layout.update(
    title = 'Amount of transactions (in $) distribution between inliers and outliers',
  xaxis_title = 'Inliers (0) vs Outliers', 
  yaxis_title = 'Amount of transactions in $')
  fig.write_image(output_path + f'amount_box_{account_id}.png')


def make_balance_line(account_id, output_path, df_cleaned_acc):
  fig1 = px.line(df_cleaned_acc, x = 'date', y = 'balance')
  fig = go.Figure(fig1.data)
  fig.layout.update(
    title = 'Balance over time',
  xaxis_title = 'Date', 
  yaxis_title = 'Balance in $')
  fig.write_image(output_path + f'balance_line_{account_id}.png')


if __name__ == "__main__":
  application.run(host = '0.0.0.0', debug=True)


