# Ibotta Fouthbrain Partner Project

Ibotta is a free cash-back and rewards payments service in agreement with about 500.000 different US and German retail stores, hosting more than 1,500 brands and saving more than $600 million to users collectively. The cash-back service is offered through an app that makes use of digital coupons to reward clients’ purchases over groceries, electronics, home and office supplies, and many other products by releasing money earnings. Even though the service is successful and leaves many clients satisfied with their buyings, Ibotta's investigation team has discovered that some clients have suspicious behaviour and might take advantage of their app to get access to cash-back in a fraudulent way. 

Given that Ibotta is growing and currently handling very large amounts of money, the idea that inspired this project is to build a model that enables the monitoring of whole transactions and successfully flagging anomalous activity with respect to the average users’ trustworthy behaviour. The machine learning product subject of this work is fully deployed as a web application that may allow Ibotta’s investigators to get access to:
    • the account details, infos that Ibotta already has about the user (e.g. the name, the balance of their account, their buyings, etc…),
    • the fraudulent behaviour risk analysis, including their risk factor (of being fraudulent) and graphs that relate the specific user to the whole users on the main features that helped in producing the risk factor (e.g. the amount of spent money, the number of days from last transaction, the date of the transaction, etc...),
    • other graphs and statistics.

This project’s ultimate objective is to increase Ibotta’s effectiveness in detecting malicious activity with an automated machine learning product that can help save money to the company and time to Ibotta’s investigators.

## The Berka Dataset 

The Berka dataset is a collection of anonymized financial details from a Czech bank for 5.300 clients and about 1.000.000 transactions. Every detail is stored into different dedicated tables that include both static and dynamic kinds of information.

Static information include details on the client’s...
    • identity (gender, birth date, whether the client shares his/her account with another person),
    • district of residence and demographics (district name, region, number of inhabitants, number of municipalities, number of cities, average salary, unemployment rate, percentage of entrepreneurs, crime rate).

Dynamic information include details on the client’s:
    • transactions history (account id, date of transaction, type, kind of operation, amount of money, balance after transaction, reason and bank/account partner),
    • issued orders (account id, recipient bank, recipient account, amount, reason),
    • active and past loans (account id, overall amount of loan, duration of loan, monthly payments on loan, loan status),
    • credit cards (card type, date of issuance).

The specific piece of data that was used to train the machine learning model is contained into the ‘trans’ table, which included all clients’ banking transactions with the fields that can be observable in the table on the right.

## Ethical Considerations 

The provided database is public, hence data has been anonymized in such a way so that every Personally Identifiable Information was removed in order to avoid going back to the real identity of the bank clients. Still it contains some personal information regarding the clients (date of birth and sex) and the district they reside into, along with some demographics about it, namely: district name, region, number of inhabitants, number of municipalities, number of cities, ratio of urban inhabitants, average salary, unemployment rate for year 1995 and 1996, number of enterpreneurs per 1000 inhabitants and number of crimes for year 1995 and 1996.

Nonetheless, we are using none of these information to learn on the model, and the only potential field that could cause bias into the model (the bank field) has been removed for too high missing values. The other information are only displayed and left for bias only if human allows to through their own judgement.

## About the web app and instructions to use it 

This tool assists the fraud investigations team in the following ways:
- View the accounts that are highly likely to have fraudulent activities in the complete dateset
- Visualize and analyze individual account's behavior
- Download raw data for individual accounts for further investigation
- Update transactions that have been falsely labeled by the Anomaly Detection model. The model learns and betters itself from this manual labelling

## Get Started with the App

- / - Home page - Scope of Work and Getting Started
- /risk - List of accounts with their corresponding risk percentage (number of anomalies per transaction)
- /investigate/(account_id) - Vizualization for analysis of the individual input account
- /data/(account_id) - Download a csv file with all transactions pertaining to the input account
- /change - Mark a particular transaction as anomaly (label = 1) or normal (label = 0) for ML model enhancement


