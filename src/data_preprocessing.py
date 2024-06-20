from scipy.signal import butter, lfilter
import numpy as np
import pandas as pd

MAPPED_FEATURES = ['gravityx', 'gravityy', 'gravityz', 'useraccelerationx', 'useraccelerationy', 'useraccelerationz']
FEATURES = ['Gravity X', 'Gravity Y', 'Gravity Z', 'User Acceleration X', 'User Acceleration Y', 'User Acceleration Z']

FS = 200
ORDER = 2
CUTOFF = 30
LOOKBACK = 20

# Filter data with lowpass-filter butterworth
def butter_filter(data, cutoff, fs, order=5, filter_type='low'):
    nyq = 0.5 * fs
    if isinstance(cutoff, list):
        normal_cutoff = [c / nyq for c in cutoff]
    else:
        normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype=filter_type, analog=False)
    y = lfilter(b, a, data)
    return y

# Preprocess data
def preproc(data):
    transformed = []
    df = data[FEATURES]
    
    buffer = [df.iloc[0].values]
    for i in range(1, len(df)):
        if not np.array_equal(df.iloc[i].values, df.iloc[i-1].values):
            buffer.append(df.iloc[i].values)
    buffer = np.array(buffer)

    df_buff = pd.DataFrame(buffer, columns=FEATURES)

    for column in FEATURES:
        if 'User Acceleration' in column:
            df_buff[column] = butter_filter(df_buff[column], CUTOFF, FS, ORDER)
        for i in range(LOOKBACK):
            df_buff[f'{column} (t-{i+1})'] = df_buff[column].shift(i+1)
    df_buff_clean = df_buff.dropna().reset_index(drop=True)
    transformed.append(df_buff_clean)
    df_cleaned = pd.concat(transformed, ignore_index=True)
    return df_cleaned