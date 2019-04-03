import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from pandas.compat import StringIO, BytesIO
import dask.dataframe as dd
import sklearn

# chunksize = 1000
#
file = './venv/train.csv'
#
# df = pd.read_csv(file, chunksize = chunksize, nrows=100_0000) # test
#
# from collections import defaultdict
# # default value of int is 0 with defaultdict
# continent_dict = defaultdict(int)
# list_app =[]
# for gm_chunk in pd.read_csv(file,chunksize=chunksize):
#     for c in gm_chunk['app']:
#         list_app.append(c)
#         if(len(list_app) == 10000):
#             break
#
# print(list_app)
# print(continent_dict)
# pd.read_csv(file, nrows=5) # Testing the file name is correct.
df = dd.read_csv(file)
# df = dd.read_csv(file)
print(df.head())
# print(len(df))
df2 = df[df.is_attributed == 1]
print(df2.head())
'''
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier()
clf.fit(df.ip)
'''
from sklearn.feature_selection import VarianceThreshold

for col in df:
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    sel.fit_transform(df.col)
# df2 = df[df.y == 'a'].x + 1

# csv_database = create_engine('sqlite:///csv_database.db')
#

# i = 0
# j = 1
# for df in pd.read_csv(file, chunksize=chunksize, iterator=True,nrows =20000):
#       df = df.rename(columns={c: c.replace(' ', '') for c in df.columns})
#       df.index += j
#       i+=1
#       df.to_sql('table', csv_database, if_exists='append')
#       j = df.index[-1] + 1
#
# df = pd.read_sql_query('SELECT * FROM table', csv_database)
# df = pd.read_sql_query('SELECT ip, app FROM table', csv_database)
# print(df)
# data_df = pd.read_csv(file,chunksize= 2000,nrows =20000)
# # print("Boston housing dataset has {} data points with {} variables each.".format(*data_df.shape))
# whole_df = []
# count = 0
# for df in data_df:
#     count += 1
#     # print (count)
#     # print("1")
#     print(df.head(5))
#     # print(df.loc[1978])
#     # print(type(df), df.shape)
#     #   whole_df.append = df
#
# # import pandas as pd
# # reader = pd.read_csv('./venv/train.csv', iterator=True, nrows =20000)
# # loop = True
# # chunkSize = 1000
# # chunks = []
# # while loop:
# #     try:
# #         chunk = reader.get_chunk(chunkSize)
# #         chunks.append(chunk)
# #     except StopIteration:
# #         loop = False
# #         print("Iteration is stopped.")
# # df = pd.concat(chunks, ignore_index=True)