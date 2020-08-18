import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.basemap import Basemap
from scipy.spatial import KDTree


def parse_csv_galveston(csv_path, simple=True):
    """
    Parse CSV and store in Pandas DataFrame
    """
    print('Reading galveston CSV...')
    df = pd.read_csv(csv_path,
                     header=0,
                     usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17],
                     parse_dates=[3, 10, 12],
                     infer_datetime_format=True,
                     dtype={
                         'vessel_id':str,
                         'latitude':np.float32,
                         'longitude':np.float32,
                         'received_time_utc':str,
                         'speed':np.float32,
                         'course':np.float32,
                         'heading':np.float32,
                         'draught':np.float32,
                         'provider_id':str,
                         'ais_type':str,
                         'added_at':str,
                         'added_by':str,
                         'updated_at':str,
                         'updated_by':str,
                         'old_last_representative':np.bool,
                         'point':str,#binary type
                         'new_navigational_status':str})
    if simple:
        df = df[['vessel_id', 'latitude', 'longitude', 'received_time_utc', 'speed', 'course', \
                 'heading', 'draught', 'new_navigational_status']]
    return df

def resample_time(df, start_time, end_time, freq, reindex=True):
    """
    Resample data every 10 min (interpolate or forward fill)
    """
    resampled_df = pd.DataFrame()
    index = pd.date_range(start=start_time, end=end_time, freq=freq)
    for vessel_id, group in df.groupby('vessel_id'):
        df_vessel = group.sort_values(['received_time_utc'], ascending=[True])
        df_vessel.index = df_vessel['received_time_utc']
        if reindex:
            union_index = index.union(df_vessel.index)
            df_vessel = df_vessel.reindex(union_index)
            df_vessel_interpolate = df_vessel[['latitude', 'longitude', 'received_time_utc', \
                'speed', 'course', 'heading', 'draught']].interpolate(method='time')
            df_vessel_ffill = df_vessel[['vessel_id', 'new_navigational_status']]\
                .fillna(method='ffill')
            df_vessel = pd.concat([df_vessel_interpolate, df_vessel_ffill], axis=1)
            df_vessel['vessel_id'] = df_vessel['vessel_id'].fillna(vessel_id)
            df_vessel = df_vessel.reindex(index)
            df_vessel['received_time_utc'] = df_vessel.index
        resampled_df = pd.concat((resampled_df, df_vessel))
    return resampled_df

