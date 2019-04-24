import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dtype = {
    'ip' :'uint32',
    'app' :'uint16',
    'device': 'uint16',
    'os' :'uint16',
    'channel': 'uint16',
    'is_attributed': 'uint8',
    'click_id': 'uint32',
}


def group_label(df, group_cols):
    for i, cols in enumerate(group_cols):
        col_name = "_".join(group_cols)
        print(i, col_name)
        group_idx = df.drop_duplicates(cols)[cols].reset_index()
        group_idx.rename(columns={'index': col_name}, inplace=True)
        df = df.merge(group_idx, on=cols, how='left')
        del group_idx
        gc.collect()
    return df


def count_agg(df, group_cols):
    for i, cols in enumerate(group_cols):
        col_name = "_".join(cols) + '_count'
        print(i, col_name)
        count = df.groupby(cols).size().reset_index(name=col_name)
        df = df.merge(count, on=cols, how='left')
        del count
        gc.collect()
    return df


def variance(df, group_cols):
    for i, cols in enumerate(group_cols):
        col_name = "_".join(cols) + '_var'
        print(i, col_name)
        temp = df.groupby(cols)[['hour']].var().reset_index().rename(index=str, columns={'hour':col_name})
        df = df.merge(temp, on =cols, how='left')
        del temp
        gc.collect()
    return df


def count_cum(df, group_cols):
    for i, cols in enumerate(group_cols):
        col_name = "_".join(cols) + '_countAccum'
        print(i, col_name)
        df[col_name] = df.groupby(cols).cumcount()
        gc.collect()
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
    return df


def next_click(df, group_cols):
    for i, cols in enumerate(group_cols):
        col_name = "_".join(cols) + '_nextClick'
        print(i, col_name)
        df[col_name] = (df.groupby(cols).click_time.shift(-1) - df.click_time).astype(np.float32)
        gc.collect()
    return df


def frequence(df, group_cols):
    for i, cols in enumerate(group_cols):
        col_name = "_".join(cols) + '_nextClick'
        print(i, col_name)
        clickFreq = df.groupby(cols)[col_name].mean().dropna().reset_index(name=("_".join(cols) + '_clickFreq'))
        df = df.merge(clickFreq, on=cols, how='left')
        del clickFreq
        gc.collect()
    return df


def time_features(df):
    df['hour'] = pd.to_datetime(df.click_time).dt.hour.astype('uint8')
    df['day'] = pd.to_datetime(df.click_time).dt.day.astype('uint8')
    df['minute'] = pd.to_datetime(df.click_time).dt.minute.astype('uint8')
    df['second'] = pd.to_datetime(df.click_time).dt.second.astype('uint8')
    df['in_test_hh'] = (3 - 2 * df['hour'].isin([4, 5, 9, 10, 13, 14])  # most frequent
                        - 1 * df['hour'].isin([6, 11, 15])).astype('uint8')  # least frequent
    # Dataframe['frequency_hour'] = Dataframe['hour'].value_counts().sort_index()
    # print(Dataframe.columns)
    print(df.head())
    print('done')
    gc.collect()
    return df


def time_frequence(df):
    frequent_hour = df.hour.value_counts().sort_index()
    frequent_day = df.day.value_counts().sort_index()
    frequent_minute = df.minute.value_counts().sort_index()
    frequent_second = df.second.value_counts().sort_index()
    # print("Frequent hours")
    # print(frequent_hour)
    # print("Frequent days")
    # print(frequent_day)
    # print("Frequent minutes")
    # print(frequent_minute)
    # print("Frequent seconds")
    # print(frequent_minute)
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
    # plt.show()


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
        ['ip'],  # 3.03
        ['channel'],
        ['os'],
        ['ip', 'device'],  # 9.88
        ['day', 'hour', 'app'],  # 4.08
        ['app', 'channel'],  # 2.8
        ['ip', 'day', 'in_test_hh'],  # 1.74
        ['ip', 'day', 'hour'],  # 0.52
        ['os', 'device'],  # 0.44
        ['ip', 'os', 'day', 'hour'],  # 0.41
        ['ip', 'device', 'day', 'hour'],  # 0.31
        ['ip', 'app', 'os']  # 0.21
    ]

    countUniq_combinations = [
        [['app'],'ip'],
        [['app', 'device', 'os', 'channel'], 'ip'],
        [['ip'], 'channel'],  # 0.9
        [['ip'], 'app'],  # 1.3
        [['ip'], 'os']  # 0.45
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
    # df['click_time'] = (df['click_time'].astype(np.int64) // 10 ** 9).astype(np.int32)
    df['click_time'] = (df['click_time'].astype(np.int64))
    df = next_click(df, nextClick_combinations)
    df = frequence(df, freq_combinations)

    # df.drop(['ip', 'click_time', 'day', 'in_test_hh'], axis=1, inplace=True)
    gc.collect()
    return df



# train: (184903890, 7)
# test: (18790469, 7)
train_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
train_df = pd.read_csv('train_sample.csv', dtype=dtype, usecols=train_cols, parse_dates=['click_time'])

# test_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id']
# # using test_supplement
# test_df = pd.read_csv('data/test_supplement.csv', dtype=dtype, usecols=test_cols, parse_dates=['click_time'])
#
# common_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time']
# all_df = pd.concat([train_df[common_cols], test_df[common_cols]])
# time_features(train_df)
# time_frequence(train_df)
# generate data
all_df = generate_features(train_df)
gc.collect()
print(all_df.info())
# train_df.to_csv(path_or_buf="ff.csv")
# train_features = all_df.iloc[:train_df.shape[0]]
#
#
# print(train_df.head())
