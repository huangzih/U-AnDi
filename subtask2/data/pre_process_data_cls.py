import numpy as np
import pandas as pd
import argparse
import gc
import os
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser()
arg = parser.add_argument
arg('--l', type=int)
args = parser.parse_args()

l = args.l

file_path = './pp_data/'
if not os.path.exists(file_path):
    os.makedirs(file_path)

filename = './origin_data/data-cls-2d-{}.csv'.format(l)
output = file_path + 'data-cls-2d-{}-pp.csv'.format(l)

df_test = pd.read_csv(filename)

def generate_mask(df):
    import numpy as np
    l = int(df['length'])
    N = int(df['N_diffmodel'])
    label = [float(i) for i in df['label'].split(',')]
    changepoint = [int(i) for i in df['changepoint'].split(',')]
    model_ls = []
    for i in range(N):
        model_ls.append(int(label[2*i]))
    mask = np.ones(l, dtype=int)*model_ls[-1]
    for i in range(N-1):
        if i == 0:
            mask[:changepoint[i]] = model_ls[i]
        else:
            mask[changepoint[i-1]:changepoint[i]] = model_ls[i]
    return ' '.join([str(i) for i in mask])

def normalizex(df):
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    data = np.array([[float(i)] for i in df['pos_x'].split(',')])
    scaler = StandardScaler() 
    scaler.fit(data)
    data2 = scaler.transform(data)
    data2 = data2.reshape(-1)
    return ','.join([str(round(i,6)) for i in data2])

def normalizey(df):
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    data = np.array([[float(i)] for i in df['pos_y'].split(',')])
    scaler = StandardScaler() 
    scaler.fit(data)
    data2 = scaler.transform(data)
    data2 = data2.reshape(-1)
    return ','.join([str(round(i,6)) for i in data2])

df_test['pos_x'] = df_test.apply(normalizex, axis=1)
df_test['pos_y'] = df_test.apply(normalizey, axis=1)

df_test['mask'] = df_test.apply(generate_mask, axis=1)

df_test.to_csv(output, index=False)
