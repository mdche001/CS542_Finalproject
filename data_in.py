import numpy as np
import pandas as pd
from pandas.compat import StringIO, BytesIO

data_df = pd.read_csv("./venv/train.csv",chunksize= 2000,nrows =20000)
# print("Boston housing dataset has {} data points with {} variables each.".format(*data_df.shape))
whole_df = []
count = 0
for df in data_df:
    count += 1
    print (count)
    print("1")
    print(df)
    # print(df.loc[1978])
    print(type(df), df.shape)
    whole_df.append = df

# import pandas as pd
# reader = pd.read_csv('./venv/train.csv', iterator=True, nrows =20000)
# loop = True
# chunkSize = 1000
# chunks = []
# while loop:
#     try:
#         chunk = reader.get_chunk(chunkSize)
#         chunks.append(chunk)
#     except StopIteration:
#         loop = False
#         print("Iteration is stopped.")
# df = pd.concat(chunks, ignore_index=True)