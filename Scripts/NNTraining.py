import numpy as np
import pandas as pd

from settings import *
from Functions.Utils import train_validate_test_split, StandardScaler_np, log_message


def load_dataset(standardize=True, corr_out=False, which_='fund_char', reuse=True, random_seed_=None):
    if os.path.exists(char_file_triple.split(".csv")[0] + "_{}.csv".format(
            which_)) and reuse:  # if we have already defined triplet of the sample
        log_message("NNTraining - load existing file...")
        df = pd.read_csv(char_file_triple.split(".csv")[0] + "_{}.csv".format(which_))
        df['Month'] = pd.to_datetime(df['Month'])
        df = df[df['Month'] >= '1995-05-31'].reset_index(drop=True).copy(deep=True)
    else:  # if not, create our own
        log_message("NNTraining - create new file from scratch...")
        df = pd.read_csv(char_file)

        if which_ == 'fund_char':
            df = df[fund_char_variables].copy(deep=True)
        elif which_ == 'holdings_char':
            df = df[holdings_char_variables].copy(deep=True)

        elif which_ == 'all':
            pass
        else:
            print("ERROR - undefined dataset, stop!")
            assert False

        df_time = df[['Month']].drop_duplicates().reset_index(drop=True).copy(deep=True)

        df_train, df_dev, df_test = train_validate_test_split(df_time, train_percent=0.3333, validate_percent=0.3333,
                                                              chronological=chronological_flag, seed=random_seed_)

        df_train['sample_id'] = 0.0
        df_dev['sample_id'] = 1.0
        df_test['sample_id'] = 2.0

        df_time = pd.concat([df_train, df_dev, df_test], axis=0)
        df_time_ = df_time.copy(deep=True)
        for i in range(3):
            df_training = df_time[df_time['sample_id'] != i].reset_index(drop=True).copy(deep=True)

            # this is still the time df, not the full training/dev/test df
            df_train, df_dev, df_dev2 = train_validate_test_split(df_training, train_percent=0.75, validate_percent=0.2,
                                                                  chronological=chronological_flag, seed=random_seed_)
            df_dev = pd.concat([df_dev, df_dev2], axis=0).reset_index(drop=True)

            df_train['sample_id_{}'.format(i)] = 0.0
            df_dev['sample_id_{}'.format(i)] = 1.0
            df_training = pd.concat([df_train, df_dev], axis=0).drop(columns=['sample_id'])

            df_time_ = pd.merge(df_time_, df_training, on=['Month'], how='left')

        df = pd.merge(df, df_time_, on=['Month'], how='inner')
        df_ = load_macro()
        df['Month'] = pd.to_datetime(df['Month']) + MonthEnd(0)
        df = pd.merge(df, df_, on=['Month'], how='inner')
        df.to_csv(char_file_triple.split(".csv")[0] + "_{}.csv".format(which_), index=False)

        df = df[df['Month'] >= '1995-05-31'].reset_index(drop=True).copy(deep=True)

    # rank standardization
    if standardize:
        df_collect = pd.DataFrame()
        for month in tqdm(df['Month'].drop_duplicates().to_list()):
            df_temp = df[df['Month'] == month].copy(deep=True)
            variables_ = df.columns
            for variable in variables_:
                if variable not in ['FundId', 'Month', 'future_ab_monthly_return'] and 'sample_id' not in variable:
                    df_temp[variable] = df_temp[variable].rank(pct=True) - 0.5
                    if variable in holdings_char_variables:
                        if variable[-2] == '_' and variable[-1].isnumeric():
                            # categorical holdings variables
                            df_temp[variable[:-2] + "_missingflag"] = pd.isna(df_temp[variable]).astype(int).astype(
                                float)
                        else:
                            # non-categorical holdings variables
                            df_temp[variable + "_missingflag"] = pd.isna(df_temp[variable]).astype(int).astype(float)
                        df_temp[variable] = df_temp[variable].fillna(0.0)
                        df_temp = df_temp.copy(deep=True)

            df_collect = pd.concat([df_collect, df_temp], axis=0)
        df = df_collect.copy(deep=True)

    # corr analysis
    if corr_out:
        df_corr = df.drop(columns=['FundId', 'Month']).corr()
        df_corr.to_excel(intermediate_directory + "corr_analysis_{}.xlsx".format(int(time.time())), index=True)

    return df


def univariate_dist():
    df, _ = load_dataset(standardize=False)
    for variable in df.columns:
        if variable not in ['FundId', 'Month'] and 'sample_id' not in variable:
            df[variable].plot.kde(bw_method=0.05, label=variable)
            plt.xlim(df[variable].quantile(0.0001), df[variable].quantile(0.9999))
            plt.legend()
            plt.savefig(intermediate_directory + "{}_dist.png".format(variable))
            plt.show()
    return None


def univariate_perc_by_month():
    df, _ = load_dataset(standardize=False)
    for variable in df.columns:
        if variable not in ['FundId', 'Month'] and 'sample_id' not in variable:
            df.groupby("Month")[variable].median().sort_index().plot(label='{}: median'.format(variable))
            df.groupby("Month")[variable].apply(lambda x: np.percentile(x, 25)).sort_index().plot(
                label='{}: 25-th'.format(variable))
            df.groupby("Month")[variable].apply(lambda x: np.percentile(x, 75)).sort_index().plot(
                label='{}: 75-th'.format(variable))
            plt.legend()
            plt.savefig(intermediate_directory + "{}_bymonth.png".format(variable))
            plt.show()
    return None


def load_training_test(df, ind=0):
    return df[df['sample_id'] != ind].reset_index(drop=True), df[df['sample_id'] == ind].reset_index(drop=True)


def attach_training_dev(nn_helper, ind=0, corr_out=False):
    if holdings_char_flag:
        df = load_dataset(standardize=True, corr_out=corr_out, which_='all')
    else:
        df = load_dataset(standardize=True, corr_out=corr_out)

    df_training, df_test = load_training_test(df, ind)
    df_train = df_training[df_training['sample_id_{}'.format(ind)] == 0.0].copy(deep=True)
    df_dev = df_training[df_training['sample_id_{}'.format(ind)] == 1.0].copy(deep=True)

    # drop sample indicators
    cols = [_ for _ in df.keys() if 'sample_id' in _]
    df_train.drop(columns=cols, inplace=True)
    df_dev.drop(columns=cols, inplace=True)
    df_test.drop(columns=cols, inplace=True)

    nn_helper.train_x_full = df_train.drop(columns=['future_ab_monthly_return']).copy(deep=True)
    nn_helper.train_x = df_train.drop(columns=['FundId', 'Month', 'future_ab_monthly_return']).copy(deep=True)
    nn_helper.train_y_full = df_train[['FundId', 'Month', 'future_ab_monthly_return']].copy(deep=True)
    nn_helper.train_y = df_train[['future_ab_monthly_return']].copy(deep=True)

    nn_helper.dev_x_full = df_dev.drop(columns=['future_ab_monthly_return']).copy(deep=True)
    nn_helper.dev_x = df_dev.drop(columns=['FundId', 'Month', 'future_ab_monthly_return']).copy(deep=True)
    nn_helper.dev_y_full = df_dev[['FundId', 'Month', 'future_ab_monthly_return']].copy(deep=True)
    nn_helper.dev_y = df_dev[['future_ab_monthly_return']].copy(deep=True)

    nn_helper.test_x_full = df_test.drop(columns=['future_ab_monthly_return']).copy(deep=True)
    nn_helper.test_x = df_test.drop(columns=['FundId', 'Month', 'future_ab_monthly_return']).copy(deep=True)
    nn_helper.test_y_full = df_test[['FundId', 'Month', 'future_ab_monthly_return']].copy(deep=True)
    nn_helper.test_y = df_test[['future_ab_monthly_return']].copy(deep=True)
    return nn_helper


def load_macro():
    """Section 1: Load CFNAI"""
    df = pd.read_excel(input_directory + char_file_macro_cfnai, sheet_name='data', usecols=['Date', 'CFNAI'])
    df.rename(columns={"Date": 'Month'}, inplace=True)

    """Section 2: Load Sentiment"""
    df_temp = pd.read_excel(input_directory + char_file_sentiment, sheet_name='DATA', usecols=['yearmo', 'SENT'])
    df_temp['Month'] = df_temp['yearmo'].apply(lambda x: str(x) + '01')
    df_temp['Month'] = pd.to_datetime(df_temp['Month']) + MonthEnd(0)
    df_temp = df_temp[['Month', 'SENT']]
    df = pd.merge(df, df_temp, on=['Month'], how='inner').dropna()

    """Section 3: Load FISD"""
    df_temp = pd.read_csv(input_directory + char_file_fisd_longterm)
    df_temp.rename(columns={'OFFERING_MONTH': 'Month'}, inplace=True)
    df_temp['Month'] = pd.to_datetime(df_temp['Month'])
    df_temp['long_term_issuance'] = df_temp[['1 Year +', '2 Year +', '3 Year +', '4 Year +', '5 Year +']].mean(axis=1)
    df = pd.merge(df, df_temp[['Month', 'long_term_issuance']], on=['Month'], how='inner').dropna()
    df_temp = pd.read_csv(input_directory + char_file_fisd_highyield, usecols=['OFFERING_MONTH', 'ratio'])
    df_temp.rename(columns={'OFFERING_MONTH': 'Month'}, inplace=True)
    df_temp['Month'] = pd.to_datetime(df_temp['Month'])
    df = pd.merge(df, df_temp.rename(columns={'ratio': 'high_yield_issuance'}), on=['Month'], how='inner').dropna()

    """Section 4: Load CPI"""
    df_temp = pd.read_csv(input_directory + char_file_cpi)
    df_temp['Month'] = pd.to_datetime(df_temp['DATE']) + MonthEnd(0)
    df_temp['CPIAUCSL'] = df_temp['CPIAUCSL'] / df_temp['CPIAUCSL'].shift(12) - 1
    df = pd.merge(df, df_temp[['Month', 'CPIAUCSL']], on=['Month'], how='inner').dropna()

    """Section 5: Load Inflation Exp"""
    df_temp = pd.read_excel(input_directory + char_file_inflation_exp, sheet_name='Expected Inflation')
    df_temp.rename(columns={'Model Output Date': 'Month'}, inplace=True)
    df_temp['Month'] = df_temp['Month'] + MonthEnd(0)
    # df_temp = df_temp[df_temp.columns[:16]]
    df_temp['expected_inflation_level'] = df_temp[[' 1 year Expected Inflation', ' 2 year Expected Inflation',
                                                   ' 3 year Expected Inflation', ' 4 year Expected Inflation',
                                                   ' 5 year Expected Inflation', ' 6 year Expected Inflation',
                                                   ' 7 year Expected Inflation', ' 8 year Expected Inflation',
                                                   ' 9 year Expected Inflation', ' 10 year Expected Inflation']].mean(
        axis=1)
    df_temp['expected_inflation_slope'] = df_temp[' 10 year Expected Inflation'] - df_temp[' 1 year Expected Inflation']
    df = pd.merge(df, df_temp[['Month', 'expected_inflation_level', 'expected_inflation_slope']], on=['Month'],
                  how='inner').dropna()

    """Section 6: Load G-Z spread"""
    df_temp = pd.read_csv(input_directory + char_file_spread)
    df_temp.rename(columns={'date': 'Month'}, inplace=True)
    df_temp['Month'] = pd.to_datetime(df_temp['Month']) + MonthEnd(0)
    df = pd.merge(df, df_temp[['Month', 'gz_spread']], on=['Month'], how='inner').dropna()

    # time-series normalization
    for col in df.columns:
        if col != "Month":
            df[col] = (df[col] - df[col].mean()) / df[col].std()  # the interpretation would be 1 std change
    return df


if __name__ == '__main__':
    # df_ = load_macro()
    # print("XXXXXXX")

    # univariate_dist()
    univariate_perc_by_month()
