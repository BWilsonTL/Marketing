from core import sfconnector as sf
import pandas as pd
from datetime import datetime, timedelta

term_p = 14


def query_param(query_string, term):
    return query_string.format(term=term)

mbr_query = """
SELECT
D_MBR.MEMBER_SK,
CASE
    WHEN D_MBR.FIRST_ORDER_DT > CURRENT_DATE()
    THEN 0
    ELSE 1
    END "BUYER",
 CASE WHEN D_MBR.FIRST_ORDER_DT <= CURRENT_DATE() THEN
    DATEDIFF('day',D_MBR.FIRST_ORDER_DT, D_MBR.LAST_ORDER_DT)
    ELSE 0
    END "BUY_LENGTH",
 DATEDIFF('day', D_MBR.LAST_LOGIN_DT, CURRENT_DATE()) "AGE_LAST_LOGIN",
 D_MBR.JOIN_DT,
 D_MBR.LAST_LOGIN_DT,
 D_MBR.LOGIN_CNT,
 CASE WHEN D_MBR.EMAIL_UNSUB_DT <= CURRENT_DATE() THEN 1
    ELSE 0
    END "EMAIL_UNSUB",
 D_MBR.FIRST_ORDER_DT,
 D_MBR.LAST_ORDER_DT,
 D_MBR.C_ORDER_CNT,
 D_MBR.ORDER_QTY - D_MBR.CANCEL_QTY - D_MBR.RETURN_QTY "TOTAL_ORDERS",
 D_MBR.ORDER_AMT - D_MBR.CANCEL_AMT - D_MBR.RETURN_AMT "ORDER_AMT"
 FROM ADM.D_MBR
 WHERE D_MBR.JOIN_DT > DATEADD(DAY, -{term}, CURRENT_DATE())
 AND D_MBR.FIRST_ORDER_DT > DATEADD(DAY, -{term}, CURRENT_DATE())
"""

session_qry = """
SELECT
    SD.SESSION_SK,
    SD.MEMBER_SK,
    MAX(DATE) "DATE",
    CASE WHEN SUM(SD.LOGIN_CNT) IS NULL THEN 0 ELSE SUM(SD.LOGIN_CNT) END "Login_Cnt",
    CASE WHEN SUM(SD.REG_CNT) IS NULL THEN 0 ELSE SUM(SD.REG_CNT) END "Registration_Cnt",
    CASE WHEN SUM(SD.HOME_CNT) IS NULL THEN 0 ELSE SUM(SD.HOME_CNT) END "Home_Cnt",
    CASE WHEN SUM(SD.ACCT_CNT) IS NULL THEN 0 ELSE SUM(SD.ACCT_CNT) END "Account_Cnt",
    CASE WHEN SUM(SD.SHOP_CNT) IS NULL THEN 0 ELSE SUM(SD.SHOP_CNT) END "Shop_Cnt",
    CASE WHEN SUM(SD.CART_CNT) IS NULL THEN 0 ELSE SUM(SD.CART_CNT) END "Cart_Cnt",
    CASE WHEN SUM(SD.CHECKOUT_CNT) IS NULL THEN 0 ELSE SUM(SD.CHECKOUT_CNT) END "Checkout_Cnt",
    CASE WHEN SUM(SD.ORDER_CONF_CNT) IS NULL THEN 0 ELSE SUM(SD.ORDER_CONF_CNT) END "Order_Confirmation_Cnt",
    CASE WHEN SUM(SD.TOTAL_ACTIVITY) IS NULL THEN 0 ELSE SUM(SD.TOTAL_ACTIVITY) END "Total_Activity",
    CASE WHEN SUM(SD.LOGIN_CNT) IS NULL THEN 0.0 ELSE ROUND(SUM(SD.LOGIN_CNT) /
        SUM(SD.TOTAL_ACTIVITY), 4) END "Login_pct",
    CASE WHEN SUM(SD.REG_CNT) IS NULL THEN 0.0 ELSE ROUND(SUM(SD.REG_CNT) /
        SUM(SD.TOTAL_ACTIVITY), 4) END "Registration_pct",
    CASE WHEN SUM(SD.HOME_CNT) IS NULL THEN 0.0 ELSE ROUND(SUM(SD.HOME_CNT) /
        SUM(SD.TOTAL_ACTIVITY), 4) END "Home_pct",
    CASE WHEN SUM(SD.ACCT_CNT) IS NULL THEN 0.0 ELSE ROUND(SUM(SD.ACCT_CNT) /
        SUM(SD.TOTAL_ACTIVITY), 4) END "Account_pct",
    CASE WHEN SUM(SD.SHOP_CNT) IS NULL THEN 0.0 ELSE ROUND(SUM(SD.SHOP_CNT) /
        SUM(SD.TOTAL_ACTIVITY), 4) END "Shop_pct",
    CASE WHEN SUM(SD.CART_CNT) IS NULL THEN 0.0 ELSE ROUND(SUM(SD.CART_CNT) /
        SUM(SD.TOTAL_ACTIVITY), 4) END "Cart_pct",
    CASE WHEN SUM(SD.CHECKOUT_CNT) IS NULL THEN 0.0 ELSE ROUND(SUM(SD.CHECKOUT_CNT) /
        SUM(SD.TOTAL_ACTIVITY), 4) END "Checkout_pct",
    CASE WHEN SUM(SD.ORDER_CONF_CNT) IS NULL THEN 0.0 ELSE ROUND(SUM(SD.ORDER_CONF_CNT) /
        SUM(SD.TOTAL_ACTIVITY), 4) END "Order_Confirmation_pct"
FROM(
    SELECT
        GD.SESSION_SK,
        GD.MEMBER_SK,
        MAX(GD.ACTIVITY_DT) "DATE",
        SUM(CASE ACTIVITY_AREA_NM WHEN 'Login' THEN ACTIVITY_CNT END) "LOGIN_CNT",
        SUM(CASE ACTIVITY_AREA_NM WHEN 'Registration' THEN ACTIVITY_CNT END) "REG_CNT",
        SUM(CASE ACTIVITY_AREA_NM WHEN 'Home' THEN ACTIVITY_CNT END) "HOME_CNT",
        SUM(CASE ACTIVITY_AREA_NM WHEN 'Account' THEN ACTIVITY_CNT END) "ACCT_CNT",
        SUM(CASE ACTIVITY_AREA_NM WHEN 'Shop' THEN ACTIVITY_CNT END) "SHOP_CNT",
        SUM(CASE ACTIVITY_AREA_NM WHEN 'Cart' THEN ACTIVITY_CNT END) "CART_CNT",
        SUM(CASE ACTIVITY_AREA_NM WHEN 'Checkout' THEN ACTIVITY_CNT END) "CHECKOUT_CNT",
        SUM(CASE ACTIVITY_AREA_NM WHEN 'Order Confirmation' THEN ACTIVITY_CNT END) "ORDER_CONF_CNT",
        SUM(GD.ACTIVITY_CNT) "TOTAL_ACTIVITY"
    FROM(
        SELECT
            SA.SESSION_SK,
            SA.MEMBER_SK,
            AG.ACTIVITY_AREA_NM,
            SA.ACTIVITY_CNT,
            SA.ACTIVITY_DT
        FROM ADM.A_SESSION_ACTIVITY_HRLY "SA"
        INNER JOIN ADM.D_ACTIVITY_GRP "AG"
            ON SA.ACTIVITY_GRP_ID = AG.ACTIVITY_GRP_ID
        WHERE AG.ACTIVITY_SITE_NM = 'Ruelala'
            AND SA.SESSION_SK > 0
            AND MEMBER_SK > 0
            AND SA.ACTIVITY_DT > DATEADD(DAY, -{term}, CURRENT_DATE())
        ) "GD"
        GROUP BY GD.SESSION_SK, GD.MEMBER_SK, GD.ACTIVITY_AREA_NM
    ) "SD"
    GROUP BY SD.SESSION_SK, SD.MEMBER_SK
"""


"""
Sums of the sum of counts of activity, but use mean() of the percentages to get a feel for
what users are spending most of their time doing.
"""


# Factor in the viewing data by department? (Parker's idea and it will certainly add to feature worth.)
# buyers = 12240 rows.

# Get the Member Data
member_data_raw = sf.SnowConnect(query_param(mbr_query, term_p), "ADW_PRD_DB", "ADM", "ADW_PRD_QRY_RL", "QUERY_WH").execute_query()
# check the data based on repeat actual buyers.
member_data_raw['RepeatBuyer'] = ((member_data_raw['BUYER'] == 1) & (member_data_raw['C_ORDER_CNT'] > 1))
member_data_raw['JOIN_DT'] = pd.to_datetime(member_data_raw['JOIN_DT'])
# Get only the members and their purchase date.
# TODO: Check the performance of the predictive algorithms by looking at activity of only repeat buyers
buyers = member_data_raw.loc[member_data_raw['BUYER'] == 1]
buyers = buyers[['MEMBER_SK', 'FIRST_ORDER_DT', 'JOIN_DT']]

session_data_raw = sf.SnowConnect(query_param(session_qry, term_p),
                                  "ADW_PRD_DB", "ADM", "ADW_PRD_QRY_RL", "QUERY_WH").execute_query()

# join the buyer data to the session data
buyer_session_join = pd.merge(session_data_raw, buyers, on='MEMBER_SK')
# get the data for members activity before they bought something.

buyer_session = buyer_session_join.loc[(buyer_session_join['DATE'] <= buyer_session_join['FIRST_ORDER_DT']) & (
    buyer_session_join['JOIN_DT'] > (datetime.today() - timedelta(days=term_p))
)]

# Get the fields that we care about.

summationFields = [
    'Login_Cnt',
    'Registration_Cnt',
    'Home_Cnt',
    'Account_Cnt',
    'Shop_Cnt',
    'Cart_Cnt',
    'Checkout_Cnt',
    'Order_Confirmation_Cnt',
    'Total_Activity'
]
averagingFields = [
    'Login_pct',
    'Registration_pct',
    'Home_pct',
    'Account_pct',
    'Shop_pct',
    'Cart_pct',
    'Checkout_pct',
    'Order_Confirmation_pct'
]
# Turn the strings into floats so that we can do the maths.
for f in averagingFields:
    buyer_session[f] = buyer_session[f].astype(float)

buyer_filter_sum = buyer_session.loc[:, ["MEMBER_SK"] + summationFields]
buyer_filter_avg = buyer_session.loc[:, ["MEMBER_SK"] + averagingFields]
# get the counts (view count and activity count)
groupedBuyersSum = buyer_filter_sum.groupby(['MEMBER_SK'], as_index=False).sum()
groupedBuyersAvg = buyer_filter_avg.groupby(['MEMBER_SK'], as_index=False).mean()
# get all the rest of the data together and create a unified data set for buyer conversion activity
buyerData = pd.merge(groupedBuyersAvg, groupedBuyersSum, on='MEMBER_SK')

#### get the same data for all non-buyers
nonBuyers = member_data_raw.loc[member_data_raw['BUYER'] != 1]
nonBuyers = nonBuyers[['MEMBER_SK', 'JOIN_DT']]
# get the session data for non buyers
nonBuyersSessionJoin = pd.merge(session_data_raw, nonBuyers, on='MEMBER_SK')

for f in averagingFields:
    nonBuyersSessionJoin[f] = nonBuyersSessionJoin[f].astype(float)

nonBuyerFilterSum = nonBuyersSessionJoin.loc[:, ["MEMBER_SK"] + summationFields]
nonBuyerFilterAvg = nonBuyersSessionJoin.loc[:, ["MEMBER_SK"] + averagingFields]

groupedNonBuyersSum = nonBuyerFilterSum.groupby(['MEMBER_SK'], as_index=False).sum()
groupedNonBuyersAvg = nonBuyerFilterAvg.groupby(['MEMBER_SK'], as_index=False).mean()



member_join = member_data_raw[['MEMBER_SK', 'JOIN_DT']]




# return the session data that is before the first buy date.

# aggregate the session data and return a single entry for each buyer for their behavior. Sum of counts, mean of percentages.

# pull out the non-buyer data from the session data and aggregate their behavior.

# merge the tables (buyer behavior before first purchase and non-buyer data)

# create a consolidated label of buyer / non-buyer based on the data (predictor)

# test/train splits

# run suite of models to determine what the best variation might be (k-fold it)

# cross validation

# confusion matrix

