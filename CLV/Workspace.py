import os
import snowflake.connector
import pandas as pd
import numpy as np
from ggplot import *
import lifetimes
import matplotlib as plt
from lifetimes.plotting import plot_frequency_recency_matrix
from lifetimes.plotting import plot_period_transactions
from lifetimes.plotting import plot_probability_alive_matrix
from lifetimes.plotting import plot_history_alive
from lifetimes import GammaGammaFitter


# connector for snowflake
def snowflake_query(query, uname, pwd, acct='ruelala', ijson=False):
    # Snowflake query engine
    conn = snowflake.connector.connect(
        user=uname,
        password=pwd,
        account=acct
    )
    cs = conn.cursor()
    col_list = list()
    try:
        cs.execute(query, _use_ijson=ijson)
        data_set = pd.DataFrame(cs.fetchall())
        meta = cs.description
        for ele in meta:
            col_list.append(ele[0])
        data_set.columns = col_list
    finally:
        cs.close()
    return data_set


# acquire the username and password for snowflake.
home_path = os.path.expanduser('~')
config_loc = os.path.join(home_path, 'Documents/snowflakeauth.txt')
with open(config_loc) as f:
    lines = f.readlines()
uname = lines[0].rstrip()
pwd = lines[1].rstrip()

"""
# Set the home location, get a pointer to the input_file
home_path = os.path.expanduser('~')
input_file = os.path.join(home_path, 'Documents/CLV/Data/', 'RawInput.dsv')
input_data = pd.read_csv(input_file, sep='\t', header=0, index_col=[0], engine='python')
"""
# Source Query:
query = """SELECT
  MEMBER_SK,
  C_ORDER_CNT - 1 "frequency",
  DATEDIFF('month', FIRST_ORDER_DT, LAST_ORDER_DT) "recency_m",
  DATEDIFF('week', FIRST_ORDER_DT, LAST_ORDER_DT) "recency_w",
  DATEDIFF('day', FIRST_ORDER_DT, LAST_ORDER_DT) "recency_d",
  DATEDIFF('month', FIRST_ORDER_DT, current_date) "T_m",
  DATEDIFF('week', FIRST_ORDER_DT, current_date) "T_w",
  DATEDIFF('day', FIRST_ORDER_DT, current_date) "T_d",
  DATEDIFF('month', LAST_ORDER_DT, current_date) "Last_Order_Age_m",
  DATEDIFF('week', LAST_ORDER_DT, current_date) "Last_Order_Age_w",
  DATEDIFF('day', LAST_ORDER_DT, current_date) "Last_Order_Age_d",
  ORDER_AMT - CANCEL_AMT - RETURN_AMT "monetary_value",
  FIRST_ORDER_DT,
  LAST_ORDER_DT,
  DATEADD('day', -1, current_date) "MODEL_BOUNDARY_DT",
  ORDER_QTY - CANCEL_QTY - RETURN_QTY "ITEMS_ORDERED"
FROM ADM.D_MBR
WHERE ORDER_QTY != CANCEL_QTY
AND ORDER_AMT - CANCEL_AMT - RETURN_AMT > 0
AND FRAUD_DT > current_date
"""

# get the data
input_data = snowflake_query(query, uname, pwd)
input_data['monetary_value'] = input_data['monetary_value'].astype(float)
#### trim the data - get only the members that have been active in 2016. ####
# get only members that have placed their first order in 2016.
recent_trimmed = input_data[pd.to_datetime(input_data['FIRST_ORDER_DT'], format='%Y-%m-%d') > '2016-01-01']
# get all members that have placed an order in 2016.
legacy_trimmed = input_data[pd.to_datetime(input_data['LAST_ORDER_DT'], format='%Y-%m-%d') > '2016-01-01']
# break out the fields that are needed for the first portion of the analysis. Recent = weeks, legacy = months
recent_members = recent_trimmed[['frequency', 'recency_w', 'T_w', 'monetary_value', 'Last_Order_Age_w', 'MEMBER_SK']]
legacy_members = legacy_trimmed[['frequency', 'recency_m', 'T_m', 'monetary_value', 'Last_Order_Age_m', 'MEMBER_SK']]

# plot the data of frequency, recency, T, and monetary value to see what we're dealing with.
ggplot(recent_members, aes(x='recency_w', y='monetary_value')) + geom_bar()
ggplot(recent_members, aes(x='frequency')) + geom_bar()
ggplot(recent_members, aes(x='T_w')) + geom_bar()
ggplot(recent_members, aes(x='Last_Order_Age_w', y='monetary_value')) + \
stat_smooth(geom='smooth', method='auto', fullrange=False)

ggplot(recent_members, aes(x='Last_Order_Age_w', y='pd.rolling_mean(monetary_value, 10)')) + geom_bar()


def fit_data(source_table, date_type='m'):
    f_field = source_table['frequency']
    if date_type == 'm':
        r_field = source_table['recency_m']
        t_field = source_table['T_m']
    elif date_type == 'w':
        r_field = source_table['recency_w']
        t_field = source_table['T_w']
    else:
        r_field = source_table['recency_d']
        t_field = source_table['T_d']
    data_model = lifetimes.BetaGeoFitter(penalizer_coef=0.0)
    data_model.fit(f_field, r_field, t_field)
    print(data_model)
    return data_model


def prediction(data, model, time_type, duration=1):
    f_field = data['frequency']
    if time_type == 'm':
        r_field = data['recency_m']
        t_field = data['T_m']
    elif time_type == 'w':
        r_field = data['recency_w']
        t_field = data['T_w']
    else:
        r_field = data['recency_d']
        t_field = data['T_d']
    data['predicted_purchases'] = model.conditional_expected_number_of_purchases_up_to_time(duration, f_field, r_field,
                                                                                            t_field)


# get the models
recent_fit = fit_data(recent_members, 'w')
legacy_fit = fit_data(legacy_members, 'm')

# plot the frequency/recency matrix
plot_frequency_recency_matrix(recent_fit, T=4)
plot_frequency_recency_matrix(legacy_fit)

# plot the probability that a customer is alive.
plot_probability_alive_matrix(recent_fit)

# Rank the customers from highest expected purchases in the next period to lowest.
# 't' is the time frame for prediction.  (e.g. t=1 will be for the next 1 of the prediction duration - if weeks, then within the next week.)
prediction(legacy_members, legacy_fit, 'm', 1)
prediction(recent_members, recent_fit, 'w', 1)
# sort it and return the highest ranked customers. (exploratory only)
best_customers = recent_members.sort_values(by='predicted_purchases').tail(10)

# assess the fit of the model.
plot_period_transactions(recent_fit)


# plot_period_transactions(legacy_fit)

# Customer Predictions (individual)
def predict_individual(member_sk, data, model, time_type, predicted_duration):
    individual = data.loc[member_sk]
    f_field = individual['frequency']
    if time_type == 'm':
        r_field = individual['recency_m']
        t_field = individual['T_m']
    elif time_type == 'w':
        r_field = individual['recency_w']
        t_field = individual['T_w']
    else:
        r_field = individual['recency_d']
        t_field = individual['T_d']
    return model.predict(predicted_duration, f_field, r_field, t_field)


# example of predicting a single user
# predict_individual(17246535, recent_members, recent_fit, 'w', 2)

# Estimating CLV - the pearson correlation is at 0.61693, which implies that there is a non-insignificant correlation between
# purchase frequency and purchase amount.  Might need to use a different approach.
# plot the data:
ggplot(recent_members, aes(x='Last_Order_Age_w', y='predicted_purchases', size='frequency')) + geom_point()
ggplot(recent_members, aes(x='Last_Order_Age_w', y='predicted_purchases')) + geom_bar()

# Get only the predicted future profit from repeat customers (whose frequency is greater than 0)
returning_customers = recent_members[recent_members['frequency'] > 0]
# run the GammaGammaFitter against the returning customers.
ggf = GammaGammaFitter(penalizer_coef=0)
ggf.fit(returning_customers['frequency'], returning_customers['monetary_value'])
p, q, v = ggf._unload_params('p', 'q', 'v')
print(ggf)

# estimate the average transaction value
returning_customers['Expected_profit'] = ggf.conditional_expected_average_profit(returning_customers['frequency'],
                                                                                 returning_customers['monetary_value'])
avg_cond_profit = ggf.conditional_expected_average_profit(returning_customers['frequency'],
                                                          returning_customers['monetary_value'])
pop_avg_profit = (p * v) / (q - 1)
print("Expected conditional Average profit: %s, Population Average Profit: %s, Average Profit: %s" % (
    avg_cond_profit.mean(),
    pop_avg_profit,
    returning_customers['monetary_value'].mean()))

ggplot(aes(x='monetary_value', y='Expected_profit'), data=returning_customers) + \
geom_point()

ggplot(aes(x='Expected_profit'), data=returning_customers) + \
geom_density()

# caluclate CLV.
returning_customers['CLV'] = ggf.customer_lifetime_value(recent_fit, returning_customers['frequency'],
                                                         returning_customers['recency_w'],
                                                         returning_customers['T_w'],
                                                         returning_customers['monetary_value'], time=12,
                                                         discount_rate=0.01)

# provide a final data set.
export_file = returning_customers[['MEMBER_SK', 'predicted_purchases', 'Expected_profit', 'CLV']].reset_index()

ggplot(export_file, aes(x='MEMBER_SK', y='CLV')) + \
geom_point(color='black', size=50) + \
geom_point(color="red", size=25)

ggplot(export_file, aes(x='MEMBER_SK', y='CLV')) + stat_smooth()

""" # this is useless.
# try it against all of the data - gives erroneous results.  Stick with the repeat customers only.
ggf_all = GammaGammaFitter(penalizer_coef=0)
ggf_all.fit(recent_members['frequency'], recent_members['monetary_value'])
p_a, q_a, v_a = ggf_all._unload_params('p', 'q', 'v')
pop_avg_profit_all = (p_a * v_a)/(q_a - 1)
print(ggf_all)
recent_members['Expected_Profit'] = ggf_all.conditional_expected_average_profit(recent_members['frequency'], recent_members['monetary_value'])
avg_cond_profit_all = recent_members['Expected_profit'].mean()
print("Expected conditional Average profit: %s, Population Average Profit: %s, Average Profit: %s" % (
    avg_cond_profit_all,
    pop_avg_profit,
    recent_members['monetary_value'].mean()))

ggplot(recent_members, aes(x='monetary_value', y='Expected_profit')) + geom_point()
"""

