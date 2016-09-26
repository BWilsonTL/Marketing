import os
import argparse
from core import sfconnector
import pandas as pd
import numpy as np
from lifetimes import GammaGammaFitter
from lifetimes import BetaGeoFitter
from lifetimes import ModifiedBetaGeoFitter
from lifetimes import ParetoNBDFitter

# Available models to use:
# BetaGeoFitter - new method
# ModifiedBetaGeoFitter - new method
# ParetoNBDFitter - old method from 1980's paper

"""
# Get argv values
argparser = argparse.ArgumentParser(description="Provide arguments for prediction model.")
argparser.add_argument('-t', '--time_block',
                       help='Assign the standard time grouping block',
                       required=False,
                       default='week',
                       choices=['day', 'week', 'month', 'quarter', 'year'])
argparser.add_argument('-pt', '--prediction_time',
                       help='Number of prediction blocks based on time_block argument',
                       required=True,
                       type=int)
argparser.add_argument('-dc', '--date_cohort',
                       help="Set the start date of cohorts in format: 'YYYY-mm-dd'",
                       required=True)
argparser.add_argument('-ct', '--cohort_field',
                       help="Use either LAST_ORDER_DT or FIRST_ORDER_DT for cohort selection",
                       required=False,
                       default='FIRST_ORDER_DT')
argparser.add_argument('-ua', '--user_acct',
                       help="Enter directory from home to location of username and password for Snowflake",
                       required=True)
argparser.add_argument('-d', '--debug_mode',
                       help="Enter to enable debug reporting on performance of model and data results",
                       action='store_true')
argparser.add_argument('-mt', '--model_type',
                       help="Enter the model type to use. BGD, MBGD, or PNDB",
                       choices=['BGF', 'MBGF', 'PNDB'],
                       required=True)

args = argparser.parse_args()
# finish the rest of this logic for command line configuration of this app.
"""
def query_mod(query_string, term_period='week'):
    """
    Modifiable query terms based on Snowflake standards.
    :param query_string: query block with 'term' substitutable parameters for a snowflake DATEDIFF function
    :param term_period: snowflake - validated date terms for DATEDIFF
    :return: returns the query block with a term period date element substituted.
    """
    return query_string.format(term=term_period)


def data_date_trim(dataframe, trim_field, date_format, date_value='1990-01-01'):
    """
    Subset of the main pull to restrict the epoch of the analysis to groups of buyers (cohorts) based on order dates
        and give only the fields that are needed for the core analysis of CLV.
    :param dataframe: the source dataframe
    :param trim_field: the field used (e.g. FIRST_ORDER_DT)
    :param date_format: snowflake date format string (e.g. '%Y-%m-%d')
    :param date_value: date in form YYYY-mm-dd for start of cohort activity
    :return: subset of source dataframe based on filtering conditions and necessary fields.
    """
    trimmed = dataframe[pd.to_datetime(dataframe[trim_field], format=date_format) > date_value]
    return trimmed[['frequency', 'recency', 'T', 'monetary_value', 'Last_order_age', 'MEMBER_SK']]


def model_fit(df, modeler, penalizer=0.0):
    """
    Fit the frequency, recency, and T values to a BetaGeoFitter model.
    :param df: source dataframe where the fields reside.
    :param penalizer: coefficient for applying a penalizer to the model.
    :return: the model
    """
    if modeler == "PNBD":
        mod = ParetoNBDFitter
    elif modeler == "MBGF":
        mod = ModifiedBetaGeoFitter
    else:
        mod = BetaGeoFitter
    model = mod(penalizer_coef=penalizer)
    model.fit(df['frequency'], df['recency'], df['T'])
    return model


def prediction(df, model_instance, future_term, field_result_name):
    """
    Create a field based on the prediction of the model, setting out predicted purchases for the set time-block
        future duration.
        e.g.: Data was pulled based on weeks back and a figure is desired for how much each buyer will spend
             in the next 4 weeks - function call would be:
                prediction(df, model, 4, '4_week_predicted_orders')
    :param df: the dataframe to be used for prediction
    :param model_instance: the model that was fit in the function model_fit()
    :param future_term: number of future time blocks (based on original query term ('weeks' / 'days' / 'months' / etc)
    :param field_result_name: user defined field name for the predicted order values.
    :return: the original dataframe with a new field appended with predicted orders.
    """
    df[field_result_name] = model_instance.\
        conditional_expected_number_of_purchases_up_to_time(future_term, df['frequency'], df['recency'], df['T'])
    return df


# GammaGammaFitter for returning customers.... (CLV)
def clv_calc(df, time_block, model_result, discount_rate=0.0, penalizer_coef=0):
    returning_customers = df[df['frequency'] > 0]
    ggf = GammaGammaFitter(penalizer_coef=penalizer_coef)
    ggf.fit(returning_customers['frequency'], returning_customers['monetary_value'])
    returning_customers['expected_profit'] = ggf.conditional_expected_average_profit(
        returning_customers['frequency'], returning_customers['monetary_value'])
    returning_customers['CLV'] = ggf.customer_lifetime_value(model_result,
                                                             returning_customers['frequency'],
                                                             returning_customers['recency'],
                                                             returning_customers['T'],
                                                             returning_customers['monetary_value'],
                                                             time=time_block,
                                                             discount_rate=discount_rate)
    if verbosity:
        p, q, v = ggf._unload_params('p', 'q', 'v')
        pop_avg_profit = (p * v) / (q - 1)
        avg_cond_profit = returning_customers['expected_profit'].mean()
        avg_real_profit = returning_customers['monetary_value'].mean()
        print(ggf)
        print("Expected conditional average profit: %s, Population Average Profit: %s, Average Real Profit: %s" %
              (avg_cond_profit,
               pop_avg_profit,
               avg_real_profit))
    return returning_customers

# Set the query
query = """SELECT
      MEMBER_SK,
      C_ORDER_CNT - 1 "frequency",
      DATEDIFF('{term}', FIRST_ORDER_DT, LAST_ORDER_DT) "recency",
      DATEDIFF('{term}', FIRST_ORDER_DT, current_date) "T",
      DATEDIFF('{term}', LAST_ORDER_DT, current_date) "Last_order_age",
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

# retrieve the username / pwd for snowflake ####################
username, password = sfconnector.account_details(os.path.join(os.path.expanduser('~'), 'Documents/snowflakeauth.txt'))
# Get the data, using default term of 'week'
raw_data = sfconnector.SnowConnect(query_mod(query), username, password).execute_query()
# Change the monetary value field to a float
raw_data['monetary_value'] = raw_data['monetary_value'].astype(float)
# Get the subset based on cohort age.  Default value for date_value is beginning of time. #################
buyer_data = data_date_trim(raw_data, 'FIRST_ORDER_DT', '%Y-%m-%d', '2016-06-01')


#############
model_type = 'BGF'
###############
# this will be replaced by sysargs: #############################
prediction_time = 4
verbosity = True
#################################################################

# run the model and create predictions
model_result = model_fit(buyer_data, model_type)
prediction(buyer_data, model_result, prediction_time, model_type + '_' + str(prediction_time))
returning_df = clv_calc(buyer_data, prediction_time, model_result, discount_rate=0.1)



def data_joiner(original_df, returning_df):
    clv_data = returning_df[['MEMBER_SK', 'predicted_purchases']]



"""
if __name__ == '__main__':
    #instantiate the main class

"""