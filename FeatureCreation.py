import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import seaborn as sns

dtype = {
    'ip' :'uint32',
    'app' :'uint16',
    'device': 'uint16',
    'os' :'uint16',
    'channel': 'uint16',
    'is_attributed': 'uint8',
    'click_id': 'uint32',
}

predictor=['ip','app','device','os','channel','is_attributed','click_id']

def group_label(df, group_cols):
    for i, cols in enumerate(group_cols):
        col_name = "_".join(group_cols)
        print(i, col_name)
        group_idx = df.drop_duplicates(cols)[cols].reset_index()
        group_idx.rename(columns={'index': col_name}, inplace=True)
        df = df.merge(group_idx, on=cols, how='left')
        del group_idx
        gc.collect()
        predictor.append(col_name)
    return df


def count_agg(df, group_cols):
    for i, cols in enumerate(group_cols):
        col_name = "_".join(cols) + '_count'
        print(i, col_name)
        count = df.groupby(cols).size().reset_index(name=col_name)
        df = df.merge(count, on=cols, how='left')
        del count
        gc.collect()
        predictor.append(col_name)
    return df


def variance(df, group_cols):
    for i, cols in enumerate(group_cols):
        col_name = "_".join(cols) + '_var'
        print(i, col_name)
        temp = df.groupby(cols)[['hour']].var().reset_index().rename(index=str, columns={'hour':col_name})
        df = df.merge(temp, on =cols, how='left')
        del temp
        gc.collect()
        predictor.append(col_name)
    return df


def count_cum(df, group_cols):
    for i, cols in enumerate(group_cols):
        col_name = "_".join(cols) + '_countAccum'
        print(i, col_name)
        df[col_name] = df.groupby(cols).cumcount()
        gc.collect()
        predictor.append(col_name)
    return df


def count_uniq(df, group_uniq_cols):
    for i, cols in enumerate(group_uniq_cols):
        group_cols, uniq_col = cols[0], cols[1]
        col_name = "_".join(group_cols) + '_uniq_' + uniq_col + '_countUniq'
        print(i, col_name)
        tmp = df.groupby(group_cols)[uniq_col].nunique().reset_index(name=col_name)
        df = df.merge(tmp, on=group_cols, how='left')
        del tmp
        gc.collect()
        predictor.append(col_name)
    return df


def next_click(df, group_cols):
    for i, cols in enumerate(group_cols):
        col_name = "_".join(cols) + '_nextClick'
        print(i, col_name)
        df[col_name] = (df.groupby(cols).date.shift(-1) - df.date).dt.seconds.astype(type)
        gc.collect()
        predictor.append(col_name)
    return df




# def frequence(df, group_cols):
#     for i, cols in enumerate(group_cols):
#         col_name = "_".join(cols) + '_nextClick'
#         print(i, col_name)
#         clickFreq = df.groupby(cols)[col_name].mean().dropna().reset_index(name=("_".join(cols) + '_clickFreq'))
#         df = df.merge(clickFreq, on=cols, how='left')
#         del clickFreq
#         gc.collect()
#     return df


def time_features(df):
    df['date'] = pd.to_datetime(df.click_time)
    df['hour'] = pd.to_datetime(df.click_time).dt.hour.astype('uint8')
    df['day'] = pd.to_datetime(df.click_time).dt.day.astype('uint8')
    df['minute'] = pd.to_datetime(df.click_time).dt.minute.astype('uint8')
    df['second'] = pd.to_datetime(df.click_time).dt.second.astype('uint8')
    df['in_test_hh'] = (3 - 2 * df['hour'].isin([4, 5, 9, 10, 13, 14])  # most frequent
                        - 1 * df['hour'].isin([6, 11, 15])).astype('uint8')  # least frequent
    # Dataframe['frequency_hour'] = Dataframe['hour'].value_counts().sort_index()
    # print(Dataframe.columns)
    predictor.append('date')
    predictor.append('hour')
    predictor.append('day')
    predictor.append('minute')
    predictor.append('second')
    predictor.append('om_test_hh')
    print(df.head())
    print('done')
    gc.collect()
    return df


def time_frequence(df):
    frequent_hour = df.hour.value_counts().sort_index()
    frequent_day = df.day.value_counts().sort_index()
    frequent_minute = df.minute.value_counts().sort_index()
    frequent_second = df.second.value_counts().sort_index()
    plt.figure(figsize=(15, 15))

    plt.subplot(221)
    frequent_hour.plot(kind='bar')
    plt.title("Frequent hours")
    plt.xlabel("Hours")
    plt.ylabel("Number")

    plt.subplot(222)
    frequent_day.plot(kind='bar')
    plt.title("Frequent days")
    plt.xlabel("Days")
    plt.ylabel("Number")

    plt.subplot(223)
    frequent_minute.plot(kind='bar')
    plt.xticks(np.arange(0, 69, step=10), (0, 9, 19, 29, 39, 49, 59))
    plt.title("Frequent  minutes")
    plt.xlabel("Minutes")
    plt.ylabel("Number")

    plt.subplot(224)
    frequent_second.plot(kind='bar')
    plt.xticks(np.arange(0, 69, step=10), (0, 9, 19, 29, 39, 49, 59))
    plt.title("Frequent seconds")
    plt.xlabel("Seconds")
    plt.ylabel("Number")
    plt.show()


def attributedAnalysis(df):
    var = ['day', 'hour']
    for feature in var:
        fig, ax = plt.subplots(figsize=(16, 6))
        # Calculate the percentage of target=1 per category value
        cat_perc = df[[feature, 'is_attributed']].groupby([feature], as_index=False).mean()
        cat_perc.sort_values(by='is_attributed', ascending=False, inplace=True)
        # Bar plot
        sns.barplot(ax=ax, x=feature, y='is_attributed', data=cat_perc)
        plt.ylabel('Percent of Download', fontsize=12)
        plt.xlabel(feature, fontsize=12)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.show()





def generate_features(df):
    print('generating time features...')
    time_features(df)
    gc.collect()

    group_combinations = [
        # ['app', 'device'],
        # ['app', 'channel']
    ]

    var_combination = [
        ['ip', 'app', 'channel'],
        ['ip', 'app', 'os'],
        ['ip','app','device'],
    ]

    count_combinations = [
        ['app'],
        ['ip'],
        ['channel'],
        ['os'],
        ['ip', 'device'],
        ['day', 'hour', 'app'],
        ['app', 'channel'],
        ['ip', 'day', 'in_test_hh'],
        ['ip', 'day', 'hour'],
        ['os', 'device'],
        ['ip', 'os', 'day', 'hour'],
        ['ip', 'device', 'day', 'hour'],
        ['ip', 'app', 'os']
    ]

    countUniq_combinations = [
        [['app'],'ip'],
        [['app', 'device', 'os', 'channel'], 'ip'],
        [['ip'], 'channel'],
        [['ip'], 'app'],
        [['ip'], 'os']
    ]

    nextClick_combinations = [
        ['ip', 'os'],
        ['ip', 'device', 'os'],
        ['ip', 'app', 'device', 'os'],
        ['ip', 'app', 'device', 'os', 'channel']
    ]

    freq_combinations = [
        ['ip', 'app', 'device', 'os']
    ]

    accum_combinations = [
        ['app'],
        ['ip'],
        ['day'],
        ['hour'],
        ['app']
    ]

    df = group_label(df, group_combinations)
    df = count_agg(df, count_combinations)
    df = variance(df,var_combination)
    df = count_cum(df, accum_combinations)
    df = count_uniq(df, countUniq_combinations)
    df['click_time'] = (df['click_time'].astype(np.int64)).astype(np.int32)
    df = next_click(df, nextClick_combinations)
    # df = frequence(df, freq_combinations)
    print(df.info())
    gc.collect()
    return df



train_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
# train_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time']

train_df = pd.read_csv('train.csv', dtype=dtype, usecols=train_cols, parse_dates=['click_time'], nrows=10000000)

time_features(train_df)
# time_frequence(train_df)

all_df = generate_features(train_df)
gc.collect()
print("Saving as csv file....")
all_df.to_csv(path_or_buf="ProcessedData.csv")
print("Complete")

