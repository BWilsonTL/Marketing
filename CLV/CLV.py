import sys
sys.path.append("/Users/bewilson/PycharmProjects/")
import os
import contextlib
import argparse
from core import sfconnector, s3connector
import pandas as pd
from ggplot import *
from lifetimes import GammaGammaFitter
from lifetimes import BetaGeoFitter
from lifetimes import ModifiedBetaGeoFitter
from lifetimes import ParetoNBDFitter
from lifetimes import plotting

# Available models to use:
# BetaGeoFitter - 2011 new method
# ModifiedBetaGeoFitter - 2002 new method
# ParetoNBDFitter - old method from 1980's paper
"""
Example use:
>>> python ./Marketing/CLV/CLV.py -t='week' -pt=8 -dc='2016-02-01' -ct='FIRST_ORDER_DT'
-sfc='Documents/snowflakeauth.txt' -v=True -ld='Documents/CLV/Auto' -mt='BGF' -ot 'Local'
-op='Documents/CLV/Auto/Outputfile.csv'
"""


def retrieve_args():
    # Get argv values
    argparser = argparse.ArgumentParser(description="Provide arguments for prediction model.")
    argparser.add_argument('-t', '--time_block',
                           help='Assign the standard time grouping block',
                           required=False,
                           default='week',
                           type=str,
                           choices=['day', 'week', 'month', 'quarter', 'year'])
    argparser.add_argument('-pt', '--prediction_time',
                           help='Number of prediction blocks based on time_block argument',
                           required=True,
                           type=int)
    argparser.add_argument('-dc', '--date_cohort',
                           help="Set the start date of cohorts in string format: 'YYYY-mm-dd'",
                           type=str,
                           required=True)
    argparser.add_argument('-ct', '--cohort_field',
                           help="Use either LAST_ORDER_DT or FIRST_ORDER_DT for cohort selection",
                           required=False,
                           type=str,
                           default='FIRST_ORDER_DT')
    argparser.add_argument('-sfc', '--snowflake_config_loc',
                           help="Enter location of txt file for username and password to Snowflake",
                           type=str,
                           required=True)
    argparser.add_argument('-v', '--verbosity_mode',
                           help="Enter to enable debug reporting on performance of model and data results",
                           required=False,
                           type=bool,
                           default=False)
    argparser.add_argument('-ld', '--log_dir',
                           help="Directory to store plots, graphs, and log.",
                           type=str,
                           required=False)
    argparser.add_argument('-mt', '--model_type',
                           help="Enter the model type to use. BGD, MBGD, or PNDB",
                           choices=['BGF', 'MBGF', 'PNDB'],
                           type=str,
                           required=True)
    argparser.add_argument('-ot', '--output_type',
                           help="Output type: S3 or Local",
                           type=str,
                           required=True,
                           choices=['Local', 'S3'])
    argparser.add_argument('-op', '--output_path',
                           help="Output path to save the generated file.  Local: path + filename.  S3: filename only.",
                           type=str,
                           required=True)
    # parse the arguments
    args = argparser.parse_args()

    # assign the variables to argument values.
    time_block_a = args.time_block
    prediction_time_a = args.prediction_time
    date_cohort_a = args.date_cohort
    cohort_field_a = args.cohort_field
    snowflake_config_loc = args.snowflake_config_loc
    verbosity_mode = args.verbosity_mode
    log_dir_a = args.log_dir
    model_type_a = args.model_type
    output_type_a = args.output_type
    output_loc_a = args.output_path

    return time_block_a, prediction_time_a, date_cohort_a, cohort_field_a, snowflake_config_loc, \
        verbosity_mode, log_dir_a, model_type_a, output_type_a, output_loc_a


def query_mod(query_string, term_period='week'):
    """
    Modifiable query terms based on Snowflake standards.
    :param query_string: query block with 'term' substitutable parameters for a snowflake DATEDIFF function
    :param term_period: snowflake - validated date terms for DATEDIFF
    :return: returns the query block with a term period date element substituted.
    """
    return query_string.format(term=term_period)


def data_date_trim(dataframe, trim_field, date_fmt, date_value='1990-01-01'):
    """
    Subset of the main pull to restrict the epoch of the analysis to groups of buyers (cohorts) based on order dates
        and give only the fields that are needed for the core analysis of CLV.
    :param dataframe: the source dataframe
    :param trim_field: the field used (e.g. FIRST_ORDER_DT)
    :param date_fmt: snowflake date format string (e.g. '%Y-%m-%d')
    :param date_value: date in form YYYY-mm-dd for start of cohort activity
    :return: subset of source dataframe based on filtering conditions and necessary fields.
    """
    trimmed = dataframe[pd.to_datetime(dataframe[trim_field], format=date_fmt) > date_value]
    return trimmed[['frequency', 'recency', 'T', 'monetary_value', 'Last_order_age', 'MEMBER_SK']]


def model_fit(df, modeler, penalizer=0.0):
    """
    Fit the frequency, recency, and T values to a BetaGeoFitter model.
    :param df: source dataframe where the fields reside.
    :param modeler: set the modeler type to use.
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


def clv_calc(df, t_block, model_used, logfile, discount_rate=0.0, penalizer_coef=0):
    """
    Calculate expected purchases for repeat customers, as well as CLV for repeat customers
    :param df: input data frame
    :param t_block: the prediction time frame (time_block * time_block type)
        e.g. if futures for a month are desired and the analysis time block is by 'week', then
        the time_block should be 4.
    :param model_used: the model that was fit on the main data.
    :param discount_rate: used for depreciation / model tuning
    :param penalizer_coef: model tuning
    :param  logfile: if verbosity is turned on, supply a file location to write to.
    :return: data frame of only the returning customers.
    """
    returning_customers = df[df['frequency'] > 0]
    ggf = GammaGammaFitter(penalizer_coef=penalizer_coef)
    ggf.fit(returning_customers['frequency'], returning_customers['monetary_value'])
    returning_customers['expected_profit'] = ggf.conditional_expected_average_profit(
        returning_customers['frequency'], returning_customers['monetary_value'])
    returning_customers['CLV'] = ggf.customer_lifetime_value(model_used,
                                                             returning_customers['frequency'],
                                                             returning_customers['recency'],
                                                             returning_customers['T'],
                                                             returning_customers['monetary_value'],
                                                             time=t_block,
                                                             discount_rate=discount_rate)
    if verbosity:
        p, q, v = ggf._unload_params('p', 'q', 'v')
        pop_avg_profit = (p * v) / (q - 1)
        avg_cond_profit = returning_customers['expected_profit'].mean()
        avg_real_profit = returning_customers['monetary_value'].mean()
        with open(logfile, "a") as f:
            f.write('GGF model output: \n' + str(ggf) + '\n')
            f.write("Expected conditional average profit: %s, Population Average Profit: %s, Average Real Profit: %s" %
                    (avg_cond_profit, pop_avg_profit, avg_real_profit))
    return returning_customers


def data_joiner(original_df, return_df):
    """
    Provide a means to join the modeled data back to the original data set and fill the merged frame's
        modeled fields with 0's to indicate buyer death.
    :param original_df: The original dataframe of buyers
    :param return_df: The returning customers dataframe with the modeled data
    :return: The merged modeled fields to the original dataframe source (with field filling)
    """
    clv_data = return_df[['MEMBER_SK', 'expected_profit', 'CLV']]
    output_table = pd.merge(original_df, clv_data, on='MEMBER_SK', how='outer')
    # fill the na values
    output_table['expected_profit'].fillna(0, inplace=True)
    output_table['CLV'].fillna(0, inplace=True)
    return output_table


def model_attributes(output_dir, input_df, model, pred_time, return_df):
    """
    Output graphs, information, and performance on the model
    :param output_dir: Directory to store the graphs / information
    :param input_df: The initial input buyer_data dataframe
    :param model: The fitted model: model_result
    :param pred_time: Prediction time block for creating RFM plots
    :param return_df: The modeled dataframe: returning_df
    :return: Nothing
    """
    # plot the input data information first
    in_recency_bar = ggplot(input_df, aes(x='recency', y='monetary_value')) + geom_bar()
    in_recency_bar.save(os.path.join(output_dir, 'Recency_Value_Raw.png'))
    # plot the frequency in a histogram
    in_frequency_hist = ggplot(input_df, aes(x='frequency')) + geom_bar()
    in_frequency_hist.save(os.path.join(output_dir, 'Frequency_Raw.png'))
    # plot the T attribute in a histogram
    in_t_hist = ggplot(input_df, aes(x='T')) + geom_histogram()
    in_t_hist.save(os.path.join(output_dir, 'T_raw.png'))
    # plot the last order age as a regression smoother figure
    in_last_order = ggplot(input_df, aes(x='Last_order_age', y='monetary_value')) +\
        stat_smooth(geom='smooth', method='auto', fullrange=False)
    in_last_order.save(os.path.join(output_dir, 'Last_Order_Age.png'))
    # plot the last order by the monetary value
    in_age_value = ggplot(input_df, aes(x='Last_order_age', y='monetary_value')) + geom_bar()
    in_age_value.save(os.path.join(output_dir, 'Order_Age_Value.png'))
    # plot the frequency recency matrix
    rfm_plot = plotting.plot_frequency_recency_matrix(model, pred_time)
    rfm_fig = rfm_plot.figure
    rfm_fig.savefig(os.path.join(output_dir, 'RFM.png'))
    # plot probability of alive customer
    pam_plot = plotting.plot_probability_alive_matrix(model)
    pam_fig = pam_plot.figure
    pam_fig.savefig(os.path.join(output_dir, 'PAM.png'))
    # model accuracy plot
    acc_plot = plotting.plot_period_transactions(model)
    acc_fig = acc_plot.figure
    acc_fig.savefig(os.path.join(output_dir, 'ACC.png'))
    # plot model predictions
    pred_plot = ggplot(return_df, aes(x='Last_order_age', y='predicted_purchases', size='frequency')) +\
        geom_point()
    pred_plot.save(os.path.join(output_dir, 'Last_Order_Predicted_Purchases_PNT.png'))
    # plot it another way
    pred2_plot = ggplot(return_df, aes(x='Last_order_age', y='predicted_purchases')) + geom_bar()
    pred2_plot.save(os.path.join(output_dir, 'Last_Order_Predicted_Purchases_BAR.png'))
    # plot member age (SK) by CLV
    mbr_clv_plot = ggplot(return_df, aes(x='MEMBER_SK', y='CLV')) +\
        geom_point(color='black', size=25) +\
        geom_point(color='red', size=10)
    mbr_clv_plot.save(os.path.join(output_dir, 'Member_CLV.png'))

    return None

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


def output_save(output_format, path, df):
    if output_format == "Local":
        output_location = os.path.join(os.path.expanduser('~'), path)
        df.to_csv(output_location, sep='|', encoding='utf-8')
    else:
        # TODO: test s3 dataframe write-> s3connector.aws_key_retrieve_local , s3connector.s3_dataframe_upload
        pass

if __name__ == '__main__':
    # get all of the arguments
    time_block, prediction_time, date_cohort, cohort_field, sf_config_loc, verbosity, doc_loc, \
        model_type, output_type, output_loc = retrieve_args()

    # prep for logging and charting
    if verbosity:
        if output_type == 'Local':
            logging_loc = os.path.join(os.path.expanduser('~'), doc_loc)
            if not os.path.exists(logging_loc):
                os.makedirs(logging_loc)
            # trash the log file if it exists.
            with contextlib.suppress(FileNotFoundError):
                os.remove(os.path.join(logging_loc, 'Notes.txt'))
        #TODO: set a directory path to save images to S3

    # retrieve the username / pwd for snowflake
    username, password = sfconnector.account_details(
        os.path.join(os.path.expanduser('~'), sf_config_loc))

    # Get the data, using default term of 'week'
    raw_data = sfconnector.SnowConnect(query_mod(query, time_block), username, password).execute_query()
    # Change the monetary value field to a float
    raw_data['monetary_value'] = raw_data['monetary_value'].astype(float)

    # Get the subset based on cohort age.  Default value for date_value is beginning of time. #################
    buyer_data = data_date_trim(raw_data, cohort_field, '%Y-%m-%d', date_cohort)

    # run the model and create predictions
    model_result = model_fit(buyer_data, model_type)
    prediction(buyer_data, model_result, prediction_time, 'predicted_purchases')
    returning_df = clv_calc(buyer_data, prediction_time, model_result, 'Notes.txt', discount_rate=0.1)
    merged_data = data_joiner(buyer_data, returning_df)
    if verbosity & output_type == 'Local':
        print("Saving charts to %s directory." % doc_loc)
        model_attributes(logging_loc, buyer_data, model_result, prediction_time, returning_df)
    output_save(output_type, output_loc, merged_data)

