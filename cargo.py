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


def interactive_sts_detection(df, display_ids=True):
    """
    Display vessels on an interactive, and detect STS with a rule based system
    - Trajectories of vessels use the last 3h
    - STS detection uses a history of 1h
    """
    _, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    # Draw map
    bmap = Basemap(projection='cea',
                   llcrnrlon=df['longitude'].min() - 1,
                   llcrnrlat=df['latitude'].min() - 1,
                   urcrnrlon=df['longitude'].max() + 1,
                   urcrnrlat=df['latitude'].max() + 1,
                   resolution='h')
    bmap.fillcontinents(color='#FFDDCC', lake_color='#DDEEFF')
    bmap.drawmapboundary(fill_color='#DDEEFF')
    bmap.drawcoastlines()
    # Resample data every 10min with interpolation
    # starts at midnight
    start_time = pd.Timestamp(df['received_time_utc'].min().strftime('%Y%m%d'))
     # ends at 23:59
    end_time = pd.Timestamp(df['received_time_utc'].max().strftime('%Y%m%d') + ' 23:59')
    df_10min = resample_time(df, start_time.to_pydatetime(), end_time.to_pydatetime(), '10min')
    time_index = sorted(set(df_10min.index))
    time_dic = {}
    for i, time_utc in enumerate(time_index):
        time_dic[time_utc] = i
    # Vessel trajectories
    plot_vessel = {}
    plot_vessel_id = {}
    for vessel_id in set(df['vessel_id']):
        plot_vessel[vessel_id], = plt.plot([], [], 'o-', markersize=1)
        plot_vessel_id[vessel_id] = plt.text(-1e6, -1e6, str(vessel_id), fontsize=8, clip_on=True)
    plot_circles = set()
    # Slider
    axcolor = 'lightgoldenrodyellow'
    ax_diam = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor=axcolor)
    sl_diam = Slider(ax_diam, 'time', 0, len(time_index)-1, valinit=0, valstep=1)
    history = 18 # * 10 min (for displaying)
    neighbor_duration = 6
    min_dist = 0.01 # in degrees for sts detection
    # Title
    title = ax.text(0.5, -0.1, '', bbox={'facecolor':'w', 'alpha':0.5, 'pad':5}, \
        transform=ax.transAxes, ha='center')
    # Update function
    def update(value):
        for c in plot_circles:
            c.remove()
        plot_circles.clear()
        value = int(value)
        start = time_index[max(0, value - history)]
        end = time_index[value]
        title.set_text('from {} to {}'.format(start.strftime('%H:%M (%d/%m/%Y)'), \
            end.strftime('%H:%M (%d/%m/%Y)')))

        filtered_df = df_10min.loc[(start <= df_10min.index) & (df_10min.index <= end)]
        for vessel_id, group_by_vessel_df in filtered_df.groupby('vessel_id'):
            coords = [bmap(x, y) for x, y in zip(group_by_vessel_df['longitude'], \
                group_by_vessel_df['latitude']) if not np.isnan(x) and not np.isnan(y)]

            (x, y) = zip(*coords) if coords else ([], [])
            plot_vessel[vessel_id].set_xdata(x)
            plot_vessel[vessel_id].set_ydata(y)
            if display_ids:
                plot_vessel_id[vessel_id].set_position((x[-1] if x else -1e6, y[-1] if y else -1e6))
        # find sts
        neighbor_trace = None
        if value >= neighbor_duration:
            for time in time_index[value - neighbor_duration: value+1]:
                df_at_time = filtered_df.loc[time]
                data = np.array([(x, y) for x, y in zip(df_at_time['longitude'], \
                    df_at_time['latitude']) if not np.isnan(x) and not np.isnan(y)])

                id_mapping = [i for i, x, y in zip(df_at_time['vessel_id'], \
                    df_at_time['longitude'], df_at_time['latitude']) \
                    if not np.isnan(x) and not np.isnan(y)]

                if len(data) >= 2:
                    T = KDTree(data)
                    pairs = T.query_pairs(min_dist)
                    pairs = set((id_mapping[i], id_mapping[j]) for i, j in pairs)
                    neighbor_trace = pairs if neighbor_trace is None \
                        else neighbor_trace.intersection(pairs)

        if neighbor_trace is None:
            return
        meta_init = filtered_df.loc[time_index[value - neighbor_duration]]
        meta_end = filtered_df.loc[time_index[value]]
        print('\n########## From {} to {}'.format(time_index[value - neighbor_duration] \
            .strftime('%H:%M (%d/%m/%Y)'), end.strftime('%H:%M (%d/%m/%Y)')))

        for id1, id2 in neighbor_trace:
            id1_data_init = meta_init.loc[meta_init['vessel_id'] == id1].iloc[0]
            id1_data_end = meta_end.loc[meta_end['vessel_id'] == id1].iloc[0]
            id2_data_init = meta_init.loc[meta_init['vessel_id'] == id2].iloc[0]
            id2_data_end = meta_end.loc[meta_end['vessel_id'] == id2].iloc[0]
            print('Detected STS {} - {}'.format(id1_data_end['vessel_id'], \
                id2_data_end['vessel_id']))

            print('    [{}] draught: {:.1f} --> {:.1f}    status: {} --> {}'.format( \
                id1_data_end['vessel_id'], id1_data_init['draught'], \
                id1_data_end['draught'], id1_data_init['new_navigational_status'], \
                id1_data_end['new_navigational_status']))

            print('    [{}] draught: {:.1f} --> {:.1f}    status: {} --> {}'.format( \
                id2_data_end['vessel_id'], id2_data_init['draught'], \
                id2_data_end['draught'], id2_data_init['new_navigational_status'], \
                id2_data_end['new_navigational_status']))

            x1, y1 = id1_data_end['longitude'], id1_data_end['latitude']
            circle_size, _ = bmap(x1 + min_dist, y1)
            x1, y1 = bmap(x1, y1)
            circle_size = abs(x1 - circle_size)
            x2, y2 = id2_data_end['longitude'], id2_data_end['latitude']
            x2, y2 = bmap(x2, y2)
            x = 0.5 * (x1 + x2)
            y = 0.5 * (y1 + y2)
            circle = plt.Circle((x, y), 3*circle_size, color='r', alpha=0.5)
            plot_circles.add(circle)
            ax.add_artist(circle)
            circle = plt.Circle((x, y), 30*circle_size, color='r', alpha=0.1)
            plot_circles.add(circle)
            ax.add_artist(circle)

    sl_diam.on_changed(update)
    plt.show()

