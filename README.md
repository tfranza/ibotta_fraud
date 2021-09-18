# Ibotta Fouthbrain Partner Project

The Ibotta Fourthbrain Partner Project relies on working with the Berka Dataset, which is a financial dataset containing anonymized information about clients of a Czech bank, accounts, transactions and related information. 

## The objective

One of the issues about banking services involves loans, because a bank doesn't want to lend money to clients that won't be able to pay back the loan, hence the bank wants to avoid such situations by issuing loans only to 'good clients'. In order to accomplish this, we will be trying to design and develop a model to understand whether a client is a good or bad one according to their information and behaviour inside the very same bank, stored as available data into the Berka Dataset.

## About the web app and instructions to use it 

Scope of Work

This tool assists the fraud investigations team in the following ways

View the accounts that are highly likely to have fraudulent activities in the complete dateset
Visualize and analyze individual account's behavior
Download raw data for individual accounts for further investigation
Update transactions that have been falsely labeled by the Anomaly Detection model. The model learns and betters itself from this manual labelling


Get Started

/ - Home page - Scope of Work and Getting Started
/risk - List of accounts with their corresponding risk percentage (number of anomalies per transaction)
/investigate/(account_id) - Vizualization for analysis of the individual input account
/data/(account_id) - Download a csv file with all transactions pertaining to the input account
/change - Mark a particular transaction as anomaly (label = 1) or normal (label = 0) for ML model enhancement


