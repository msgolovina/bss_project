import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
import branca
import folium
import urllib
import json
import re
import datetime


from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import mean_squared_error

API_KEY_GOOGLE = "INSERT GOOGLE API HERE"



def describe_data(dataset, data_name, n_observations, n_dropped):

    '''
    describe the data; all the days with more than 20 missing
    observations are not included in the final dataset
    '''

    print('''
    Collected {} for:
              \u2022 {} days without missing observations;
              \u2022 {} days with 1 to 20 missing values.
    {} days with more than 20 missing values were dropped.

    The data covers the period from {:%d %B %Y %H:%M} to {:%d %B %Y %H:%M}'''.format(
        data_name,
        pd.value_counts(n_observations)[720],
        pd.value_counts(n_observations[(n_observations >= 700) & (n_observations < 720)]).sum(),
        n_dropped,
        dataset.index.min().date(),
        dataset.index.max().date()))

def altitude(latitude, longitude):
    '''
    this function takes points indicating station locations and requests
    elevation of these points from Google Maps API Elevation Service;
    Google API key is AIzaSyDl8r2Vup0mrmphy8AtRD8IqV-TgOslb4A
    '''

    key = API_KEY_GOOGLE
    url = 'https://maps.googleapis.com/maps/api/elevation/json' + \
          '?locations=' + str(latitude) + ',' + str(longitude) + '&key=' + key
    request = urllib.request.urlopen(url)
    try:
        results = json.load(request).get('results')
        if len(results) > 0:
            altitude = results[0].get('elevation')
            return altitude
        else:
            print('HTTP GET Request failed')
    except ValueError:
        print('JSON decode failed: ' + str(request))


def plot_elevation_map(stations, lat, lon, station, altitude):

    '''
    plot interactive folium map of bicycle stations in Dublin;
    the colour of the station marker depends on the elevation of the station above sea level
    '''

    cm = branca.colormap.LinearColormap(['black', 'deepskyblue'], vmin = np.min(stations['Alt']), vmax = np.max(stations['Alt']))

    station_markers = folium.map.FeatureGroup()
    for x, y, name, alt in zip(lat, lon, station, altitude):
        station_markers.add_child(
            folium.features.CircleMarker(
                [x, y],
                radius=2.7,
                color = cm(alt),
                opacity=0.4,
                fill = True,
                fill_color = cm(alt),
                fill_opacity = 1,
                popup = name)
            )
    folium_map = folium.Map(location=[53.350140, -6.266155],
                            zoom_start=13,
                            tiles="CartoDB positron")
    folium_map.add_child(station_markers)
    cm.caption = 'Elevation above sea level'
    folium_map.add_child(cm)
    return folium_map

def plot_clusters_map(stations, data, clusters_array, cluster_colours):

    '''
    plot interactive folium map of bicycle stations in Dublin;
    the colour of the station marker denotes the cluster of this station
    based on its usage pattern
    '''

    try:
        stations.drop(['Cluster'], 1, inplace=True)
    except:
        pass
    cl = pd.DataFrame({'Name': data.T.index.values, 'Cluster': clusters_array})
    cl.Name = cl.Name.apply(lambda x: x.replace('_', ' ').replace("'", ""))
    stations = stations.merge(cl, on='Name', how='inner')

    station_markers = folium.map.FeatureGroup()
    for x, y, name, cluster in zip(stations.Latitude, stations.Longitude, stations.Name, stations.Cluster):
        station_markers.add_child(
            folium.features.CircleMarker(
                [x, y],
                radius=2.7,
                color=cluster_colours[cluster],
                opacity=0.4,
                fill=True,
                fill_color=cluster_colours[cluster],
                fill_opacity=1,
                popup=name)
        )
    folium_map = folium.Map(location=[53.350140, -6.266155],
                            zoom_start=13.3,
                            tiles="CartoDB positron")
    folium_map.add_child(station_markers)
    return folium_map



def plot_SSE(weekdays_norm, weekends_norm):

    '''
    plot within cluster sum of squared errors for clusterization of
    bicycle usage on weekdays and weekends;
    the appropriate number of clusters can be chosen using the elbow method
    '''

    figure, (axis_weekdays, axis_weekends) = plt.subplots(1, 2, figsize=(20, 6))

    sses = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=7).fit(weekdays_norm.T)
        clusters = kmeans.predict(weekdays_norm.T)
        sse = sum([sum(pow(weekdays_norm.T.iloc[i, :] - kmeans.cluster_centers_[clusters[i]], 2)) for i in
                   range(weekdays_norm.T.shape[0])])
        sses.append(sse)
    axis_weekdays.plot(sses, marker='.', ms=7, lw=1, color='red')
    axis_weekdays.set_xticks(range(10))
    axis_weekdays.set_xticklabels(range(1, 11))
    axis_weekdays.set_xlabel('№ of clusters')
    axis_weekdays.set_ylabel('Within cluster sum of squared errors')
    axis_weekdays.set_title('SSE by n_clusters for weekdays')
    axis_weekdays.margins(0.02)

    sses1 = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(weekends_norm.T)
        clusters = kmeans.predict(weekends_norm.T)
        sse = sum([sum(pow(weekends_norm.T.iloc[i, :] - kmeans.cluster_centers_[clusters[i]], 2)) for i in
                   range(weekends_norm.T.shape[0])])
        sses1.append(sse)
    axis_weekends.plot(sses1, marker='.', ms=7, lw=1, color='blue')
    axis_weekends.set_xticks(range(10))
    axis_weekends.set_xticklabels(range(1, 11))
    axis_weekends.set_xlabel('№ of clusters')
    axis_weekends.set_ylabel('Within cluster sum of squared errors')
    axis_weekends.set_title('SSE by n_clusters for weekends')
    axis_weekends.margins(0.02)


def plot_cluster_usage_patterns(weekdays_norm, weekends_norm, cluster_colours_wd, cluster_colours_we):

    '''
    plot average station availability for each of the clusters
    on weekdays and weekends; vertical red line marks 12:00 on both
    graphs to make it more clear how the most busy hour is different
    for weekdays and weekends
    '''

    figure, (axis_weekdays, axis_weekends) = plt.subplots(1, 2, figsize=(20, 6))

    # weekdays
    n_clusters_weekdays = 3
    model = KMeans(n_clusters=n_clusters_weekdays, random_state=7)
    model_fit = model.fit(weekdays_norm.T)
    for k, colour in zip(model_fit.cluster_centers_, cluster_colours_wd):
        axis_weekdays.plot(100 * k, color=colour, label=colour)
    axis_weekdays.set_ylim([5, 100])
    xticks = range(0, 289, 24)
    xticks_labels = ["%02d:%02d" % divmod(i * 5, 60) for i in [int(tick) for tick in xticks]]
    axis_weekdays.set_xticks(xticks)
    axis_weekdays.set_xticklabels(xticks_labels)
    axis_weekdays.set_xlabel('hour of the day')
    axis_weekdays.set_ylabel('% of the station full')
    axis_weekdays.set_title('Clusters of bike usage patterns at weekdays')
    axis_weekdays.margins(0.02)
    axis_weekdays.legend(labels=['empty at working time', 'full at working time', 'balanced'])
    axis_weekdays.axvline(x = 144, color = 'r', alpha = 0.2)
    clusters_weekdays = model_fit.predict(weekdays_norm.T)

    # weekends

    n_clusters_weekends = 3
    model = KMeans(n_clusters=n_clusters_weekends, random_state=7)
    model_fit = model.fit(weekends_norm.T)
    for k, colour in zip(model_fit.cluster_centers_, cluster_colours_we):
        axis_weekends.plot(100 * k, color=colour, label=colour)
    axis_weekends.set_ylim([25, 100])
    xticks = range(0, 289, 24)
    xticks_labels = ["%02d:%02d" % divmod(i * 5, 60) for i in [int(tick) for tick in xticks]]
    axis_weekends.set_xticks(xticks)
    axis_weekends.set_xticklabels(xticks_labels)
    axis_weekends.set_xlabel('hour of the day')
    axis_weekends.set_ylabel('% of the station full')
    axis_weekends.set_title('Clusters of bike usage patterns at weekends')
    axis_weekends.margins(0.02)
    axis_weekends.legend(
        labels=['balanced', 'empty at lunch time', 'full at lunch time'])
    axis_weekends.axvline(x=144, color = 'r', alpha = 0.2)
    clusters_weekends = model_fit.predict(weekends_norm.T)

    return clusters_weekdays, clusters_weekends


def get_weather_features(weather):

    '''
    transform weather dataset to create new numerical features from
    text description of the weather; first, dummy features for snow, rain, clouds,
    fog, thinder and hail are created. second, adjectives describing intensity
    (light, heavy) are used to add more levels to these features.
    '''
    idx = weather.index
    weather_types = pd.DataFrame(weather.Weather.value_counts()).reset_index()
    weather_types['agg_types'] = weather_types['index'].copy().apply(lambda x: re.sub('Windy|/', '', x).strip())
    weather['Wind_Speed'] = weather['Wind_Speed'].replace('calm', 0).astype(int)
    weather_types['Snow'] = [1 if 'Snow' in x else 0 for x in weather_types.agg_types]
    weather_types['Rain'] = [1 if ('Rain' in x) | ('Drizzle' in x) | ('Shower' in x) else 0 for x in
                             weather_types.agg_types]
    weather_types['Clouds'] = [1 if 'Cloudy' in x else 0 for x in weather_types.agg_types]
    weather_types['Fog'] = [1 if ('Fog' in x) | ('Mist' in x) else 0 for x in weather_types.agg_types]
    weather_types['Thunder'] = [1 if ('Thunder' in x) | ('Storm' in x) else 0 for x in weather_types.agg_types]
    weather_types['Hail'] = [1 if 'Hail' in x else 0 for x in weather_types.agg_types]

    for colname in weather_types.columns[3:].values:
        low = [0.5 if 'Light' in x else 1 for x in weather_types.agg_types]
        high = [2 if 'Heavy' in x else 1 for x in weather_types.agg_types]
        weather_types[colname] = weather_types[colname] * low * high
    weather_types = weather_types.drop(['Weather', 'index'], 1)
    weather['idx'] = weather.index
    weather['Weather'] = weather['Weather'].apply(lambda x: re.sub('Windy|/', '', x).strip())
    weather = pd.merge(weather, weather_types, how='left', left_on='Weather', right_on='agg_types').drop_duplicates()
    weather.index = idx
    del weather['agg_types'], weather['idx']

    return weather

def get_seasonal_features(features):

    '''
    add features denoting the month and the season of each observation
    '''

    features['Month'] = [x.month for x in features.index]

    season = []
    for x in features['Month']:
        if x in [12, 1, 2]:
            season.append('Winter')
        elif x in [3, 4, 5]:
            season.append('Spring')
        elif x in [6, 7, 8]:
            season.append('Summer')
        else:
            season.append('Autumn')
    features['Season'] = season

    new_feats = OneHotEncoder(sparse=False).fit_transform(
        LabelEncoder().fit_transform(features['Season'].values).reshape(-1, 1))
    for name, onehotfeat in zip(['Spring', 'Summer', 'Winter'], new_feats.T):
        features[name] = onehotfeat.astype(int)
    features.drop(['Season'], 1, inplace=True)
    return features


def plot_usage_examples(data, station_1, station_2):
    df_copy = data.copy()
    df_copy['weekday'] = df_copy.index.weekday
    weekdays = df_copy.loc[df_copy['weekday'] < 5].drop(['weekday'], 1)
    weekends = df_copy.loc[df_copy['weekday'] >= 5].drop(['weekday'], 1)

    # Group by minute of day and take average for each station
    weekdays['time'] = weekdays.index.to_series().apply(lambda x: datetime.datetime.strftime(x, '%H:%M'))
    weekdays = weekdays.groupby('time').agg('mean').astype(int)
    weekends['time'] = weekends.index.to_series().apply(lambda x: datetime.datetime.strftime(x, '%H:%M'))
    weekends = weekends.groupby('time').agg('mean').astype(int)

    # Normalise each station between 0 and 1 as each station has different capacity
    weekdays_norm = weekdays / weekdays.max()
    weekends_norm = weekends / weekends.max()

    weekends_norm.index = pd.to_datetime(weekends.index)
    weekdays_norm.index = pd.to_datetime(weekdays.index)

    wd_1 = weekdays_norm.resample('1T')[station_1].mean().interpolate()
    we_1 = weekends_norm.resample('1T')[station_1].mean().interpolate()
    wd_2 = weekdays_norm.resample('1T')[station_2].mean().interpolate()
    we_2 = weekends_norm.resample('1T')[station_2].mean().interpolate()

    figure, (ax, ax1) = plt.subplots(1, 2, figsize=(20, 6))
    ax.plot(wd_1, color='indianred')
    ax.plot(we_1, color='dodgerblue')
    ax.set_title('{} station availability'.format(station_1.capitalize().replace('_', ' ')))
    ax.set_ylabel('# of bikes (out of 30)')
    ax.set_xlabel('hour of the day')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.legend(['weekdays', 'weekends'])

    ax1.plot(wd_2, color='indianred')
    ax1.plot(we_2, color='dodgerblue')
    ax1.set_title('{} station availability'.format(station_2.capitalize().replace('_', ' ')))
    ax1.set_ylabel('station occupancy (0 to 1)')
    ax1.set_xlabel('hour of the day')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.legend(['weekdays', 'weekends'])
    plt.tight_layout()

def plot_stations_with_overdemand(data, stations):
    data['DATE'] = [x.date() for x in data.index]
    data.columns = [x.replace("'", '') for x in data.columns.values]
    dates = []
    station_was_empty_3by2mins = []
    station_was_full_3by2mins = []
    for day in data.DATE.unique():
        dates.append(day)
        df_day = data[data.DATE == day]
        counter_stations_full = 0
        counter_stations_empty = 0
        for name, capacity in zip(stations.Name, stations.Capacity):
            if df_day[name.replace(' ', '_')][df_day[name.replace(' ', '_')] == capacity].shape[0] > 45:
                counter_stations_full += 1
            if df_day[name.replace(' ', '_')][df_day[name.replace(' ', '_')] == 0].shape[0] > 45:
                counter_stations_empty += 1
        station_was_full_3by2mins.append(counter_stations_full)
        station_was_empty_3by2mins.append(counter_stations_empty)

    dff = pd.DataFrame({'full': station_was_full_3by2mins, 'empty': station_was_empty_3by2mins})
    dff.index = pd.DatetimeIndex(dates)
    dff = dff.sort_index(ascending=True)
    dff['weekday'] = [x.weekday() for x in dff.index]
    dff = dff.resample('7D').mean().bfill()
    plt.plot(dff['full'])
    plt.plot(dff['empty'])
    plt.legend(labels=['full for 1.5+ hrs', 'empty for 1.5+ hrs'])
    plt.xlabel('Month')
    plt.ylabel('Number of stations')
    plt.show()


def estimate_adjusted_rate(data, station, step_size):
    pb_weekdays = data
    diff = pb_weekdays[station].diff()
    diff[diff.isnull()] = 0
    pb_weekdays['difference'] = diff
    pb_weekdays['peak'] = pb_weekdays['upward_peak'].replace(2, 1) + pb_weekdays['downward_peak'].replace(2, 1)
    pb_weekdays['difference'] = [x if y == 0 else 0 for x, y in zip(pb_weekdays.difference, pb_weekdays.peak)]
    pb_weekdays['maxval'] = [2 if x == 30 else 0 for x in pb_weekdays[station]]
    pb_weekdays['minval'] = [2 if x == 0 else 0 for x in pb_weekdays[station]]
    pb_weekdays['sum_diffs_arr'] = [x if x > 0 else 0 for x in pb_weekdays.difference]
    pb_weekdays['sum_diffs_dep'] = [x if x < 0 else 0 for x in pb_weekdays.difference]
    pb_weekdays['valid_arr_time'] = [2 if y == 0 else 0 for y in pb_weekdays.maxval]
    pb_weekdays['valid_dep_time'] = [2 if y == 0 else 0 for y in pb_weekdays.minval]

    pb_weekdays = pd.DataFrame(pb_weekdays.resample(str(step_size)+'T').sum())
    pb_weekdays['time'] = pb_weekdays.index.to_series().apply(lambda x: datetime.datetime.strftime(x, '%H:%M'))
    pb_weekdays['arr_rate'] = pb_weekdays.sum_diffs_arr / (pb_weekdays.valid_arr_time / 60)
    pb_weekdays['dep_rate'] = pb_weekdays.sum_diffs_dep / (pb_weekdays.valid_dep_time / 60)
    pb_weekdays[pb_weekdays['arr_rate'].isnull()] = 0
    pb_weekdays[pb_weekdays['arr_rate'] == np.inf] = 0
    pb_weekdays[pb_weekdays['dep_rate'].isnull()] = 0
    pb_weekdays[pb_weekdays['dep_rate'] == np.inf] = 0
    pb_weekdays[pb_weekdays['arr_rate'] == -np.inf] = 0
    pb_weekdays[pb_weekdays['dep_rate'] == -np.inf] = 0
    diff_arr_12 = pb_weekdays.groupby('time').agg('mean')['arr_rate'][1:]
    diff_dep_12 = -pb_weekdays.groupby('time').agg('mean')['dep_rate'][1:]

    diff = data[(data.upward_peak == 0) & (data.downward_peak == 0)][station].diff()
    diff[diff.isnull()] = 0
    diff_arr_1 = pd.DataFrame(diff[diff >= 0].resample(str(step_size)+'T').sum()) * 60/step_size
    diff_dep_1 = pd.DataFrame(diff[diff <= 0].resample(str(step_size)+'T').sum()) * 60/step_size
    diff_arr_1['time'] = diff_arr_1.index.to_series().apply(lambda x: datetime.datetime.strftime(x, '%H:%M'))
    diff_arr_1 = diff_arr_1.groupby('time').agg('mean')
    diff_dep_1['time'] = diff_dep_1.index.to_series().apply(lambda x: datetime.datetime.strftime(x, '%H:%M'))
    diff_dep_1 = -diff_dep_1.groupby('time').agg('mean')
    figure, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 6))
    ax1.set_ylabel('rate (events per hour)')
    ax1.set_title('Arrival rate. Interval: {} min'.format(step_size))
    diff_arr_1[station].plot(drawstyle="steps", alpha=0.7, color = 'indianred', ax = ax1)
    diff_arr_12.plot(drawstyle="steps", alpha=0.7, color = 'dodgerblue', ax = ax1)
    ax2.set_title('Departure rate. Interval: {} min'.format(step_size))
    ax2.set_ylabel('rate (events per hour)')
    diff_dep_1[station].plot(drawstyle="steps", alpha=0.7, color = 'indianred', ax=ax2)
    diff_dep_12.plot(drawstyle="steps", alpha=0.7, color = 'dodgerblue', ax=ax2)
    return diff_arr_1, diff_arr_12, diff_dep_1, diff_dep_12



def repeat_simulation_est(estimate_adjusted_rate_sim, sim_station, ar_adj, dep_adj, n_months):
    d_true = []
    d_pred = []

    for i in range(n_months):
        realizations, inds, realizations_finite = sim_station(ar_adj, dep_adj, days_=25, silent = True)
        inds = [x for subset in inds for x in subset]
        realizations = [x for subset in realizations for x in subset]
        realizations = pd.Series(realizations)
        realizations.index = inds
        inds_finite = inds  # [x for subset in inds for x in subset]
        realizations_finite = [x for subset in realizations_finite for x in subset]
        realizations_finite = pd.Series(realizations_finite)
        realizations_finite.index = inds_finite
        realizations_finite_data = pd.DataFrame(realizations_finite)
        realizations_finite_data.columns = ['occ']
        arr_adj_hat, dep_adj_hat, _, _ = estimate_adjusted_rate_sim(realizations_finite_data, 15, silent=True)
        days = np.unique(realizations_finite.index.date)
        d_, pred_ = calc_some_true_and_preds(days, realizations_finite, estimate_adjusted_rate_sim, realizations_finite_data, realizations)
        d_true.append(d_)
        d_pred.append(pred_)

    d_pred = [x for subset in d_pred for x in subset]
    d_true = [x for subset in d_true for x in subset]
    print('Successfully calculated RMSE for {} days:'.format(n_months*29), np.sqrt(mean_squared_error(y_true=d_true, y_pred=d_pred)))
    #print('RMSE for 2 hrs horizon:' ,np.sqrt(mean_squared_error(y_true=d_true, y_pred=d_pred)), np.sqrt(mean_squared_error(y_true=hor_true_, y_pred=hor_pred_)))
    return np.sqrt(mean_squared_error(y_true=d_true, y_pred=d_pred)), d_true, d_pred#, np.sqrt(mean_squared_error(y_true=hor_true_, y_pred=hor_pred_))

def calc_some_true_and_preds(days, realizations_finite, estimate_adjusted_rate_sim, realizations_finite_data, realizations):
    k = 2
    for d in days:
        day = realizations_finite[realizations_finite.index.date == d]
        arr_adj_hat, dep_adj_hat, _, _ = estimate_adjusted_rate_sim(realizations_finite_data, 15, silent=True)

        arr_adj_hat.index = [datetime.datetime.strptime(x, '%H:%M') for x in arr_adj_hat.index]
        dep_adj_hat.index = [datetime.datetime.strptime(x, '%H:%M') for x in dep_adj_hat.index]
        arr_rate = arr_adj_hat[
            arr_adj_hat.index.time >= day.index[np.min(np.where(realizations_finite == 0))].time()]
        dep_rate = dep_adj_hat[
            dep_adj_hat.index.time >= day.index[np.min(np.where(realizations_finite == 0))].time()]
        arrival_times = []
        for hour in range(len(arr_rate)):
            n_arrivals = np.random.poisson(lam=arr_rate[hour % len(arr_rate)] * 24 / len(
                arr_rate))  # * np.max([1+np.random.normal(0, 0.5), 0.0001]))
            for x in range(n_arrivals):
                arrival_times = np.append(arrival_times, np.random.uniform() + hour)
        arrival_times = sorted(arrival_times)
        departure_times = []
        for hour in range(len(dep_rate)):
            n_departures = np.random.poisson(lam=dep_rate[hour % len(dep_rate)] * 24 / len(
                dep_rate))  # * np.max([1+np.random.normal(0, 0.5), 0.0001]))
            for x in range(n_departures):
                departure_times = np.append(departure_times, np.random.uniform() + hour)

        departure_times = sorted(departure_times)
        arr = pd.DataFrame(arrival_times)
        arr['arr'] = np.ones(len(arrival_times))
        dep = pd.DataFrame(departure_times)
        dep['arr'] = [-x for x in np.ones(len(departure_times))]
        arr = arr.append(dep).sort_values(by=0)

        start = 0
        occupancy = start + np.cumsum(arr['arr'])
        arr['occupancy'] = occupancy
        arr = arr.reset_index(drop=True)

        result = ['{0:02.0f}:{1:02.0f}'.format(*divmod(x * 60, 60)) for x in arr[0] / len(arr_adj_hat) * 24]
        for j, x in enumerate(result):
            if '60' in x:
                result[j] = x.replace('60', '59')
            if '24:' in x:
                result[j] = ('23:59')

        result = [datetime.datetime.strptime(x, '%H:%M') for x in result]
        result = [x + datetime.timedelta(hours=day.index[np.min(np.where(realizations_finite == 0))].hour, #-1
                                         minutes=day.index[np.min(np.where(realizations_finite == 0))].minute) for x
                  in result]

        arr.index = [datetime.datetime.strftime(x, '%H:%M') for x in result]
        arr.index = pd.to_datetime(arr.index)
        arr1 = arr['occupancy'].resample('2T').mean().bfill()

        arr1 = arr1.apply(lambda x: round(x, 0))
        arr1.index = arr1.index.map(lambda t: t.replace(year=2007, month=6, day=k))
        k += 1
        true_demand = realizations[(realizations.index.date == d) & (realizations.index.time >= arr1.index.time[0])]
        #horizon_2_true = realizations[(realizations.index.date == d) & (realizations.index.time == arr1.index.time[0])+datetime.timedelta(hours=2)]
        #horizon_pred =  arr1.index.time[0]+datetime.timedelta(hours=2)
        idx_ = true_demand.index
        predicted_demand = [x for x, y in zip(arr1, arr1.index) if y in true_demand.index]
        idx = [y for x, y in zip(arr1, arr1.index) if y in true_demand.index]
        predicted_demand = pd.Series(predicted_demand)
        predicted_demand.index = idx
        true_demand = [x for x, y in zip(true_demand, true_demand.index) if y in predicted_demand.index]
        true_demand = pd.Series(true_demand)
        return true_demand, predicted_demand#, horizon_2_true, horizon_pred



def calc_some_true_and_preds_1H(days, realizations_finite, estimate_adjusted_rate_sim, realizations_finite_data, realizations):
    k = 2
    for d in days:
        day = realizations_finite[realizations_finite.index.date == d]
        arr_adj_hat, dep_adj_hat, _, _ = estimate_adjusted_rate_sim(realizations_finite_data, 15, silent=True)

        arr_adj_hat.index = [datetime.datetime.strptime(x, '%H:%M') for x in arr_adj_hat.index]
        dep_adj_hat.index = [datetime.datetime.strptime(x, '%H:%M') for x in dep_adj_hat.index]
        arr_rate = arr_adj_hat[
            arr_adj_hat.index.time >= day.index[np.min(np.where(realizations_finite == 0))-datetime.timedelta(hours=1)].time()]
        dep_rate = dep_adj_hat[
            dep_adj_hat.index.time >= day.index[np.min(np.where(realizations_finite == 0))-datetime.timedelta(hours=1)].time()]
        arrival_times = []
        for hour in range(len(arr_rate)):
            n_arrivals = np.random.poisson(lam=arr_rate[hour % len(arr_rate)] * 24 / len(
                arr_rate))  # * np.max([1+np.random.normal(0, 0.5), 0.0001]))
            for x in range(n_arrivals):
                arrival_times = np.append(arrival_times, np.random.uniform() + hour)
        arrival_times = sorted(arrival_times)
        departure_times = []
        for hour in range(len(dep_rate)):
            n_departures = np.random.poisson(lam=dep_rate[hour % len(dep_rate)] * 24 / len(
                dep_rate))  # * np.max([1+np.random.normal(0, 0.5), 0.0001]))
            for x in range(n_departures):
                departure_times = np.append(departure_times, np.random.uniform() + hour)

        departure_times = sorted(departure_times)
        arr = pd.DataFrame(arrival_times)
        arr['arr'] = np.ones(len(arrival_times))
        dep = pd.DataFrame(departure_times)
        dep['arr'] = [-x for x in np.ones(len(departure_times))]
        arr = arr.append(dep).sort_values(by=0)

        start = 0
        occupancy = start + np.cumsum(arr['arr'])
        arr['occupancy'] = occupancy
        arr = arr.reset_index(drop=True)

        result = ['{0:02.0f}:{1:02.0f}'.format(*divmod(x * 60, 60)) for x in arr[0] / len(arr_adj_hat) * 24]
        for j, x in enumerate(result):
            if '60' in x:
                result[j] = x.replace('60', '59')
            if '24:' in x:
                result[j] = ('23:59')

        result = [datetime.datetime.strptime(x, '%H:%M') for x in result]
        result = [x + datetime.timedelta(hours=day.index[np.min(np.where(realizations_finite == 0))].hour, #-1
                                         minutes=day.index[np.min(np.where(realizations_finite == 0))].minute) for x
                  in result]

        arr.index = [datetime.datetime.strftime(x, '%H:%M') for x in result]
        arr.index = pd.to_datetime(arr.index)
        arr1 = arr['occupancy'].resample('2T').mean().bfill()

        arr1 = arr1.apply(lambda x: round(x, 0))
        arr1.index = arr1.index.map(lambda t: t.replace(year=2007, month=6, day=k))
        k += 1
        true_demand = realizations[(realizations.index.date == d) & (realizations.index.time >= arr1.index.time[0])]
        idx_ = true_demand.index
        predicted_demand = [x for x, y in zip(arr1, arr1.index) if y in true_demand.index]
        idx = [y for x, y in zip(arr1, arr1.index) if y in true_demand.index]
        predicted_demand = pd.Series(predicted_demand)
        predicted_demand.index = idx
        true_demand = [x for x, y in zip(true_demand, true_demand.index) if y in predicted_demand.index]
        true_demand = pd.Series(true_demand)
        return true_demand, predicted_demand


def repeat_simulation_est_1H(estimate_adjusted_rate_sim, sim_station, ar_adj, dep_adj, n_months):
    d_true = []
    d_pred = []
    for i in range(n_months):
        realizations, inds, realizations_finite = sim_station(ar_adj, dep_adj, days_=25, silent = True)
        inds = [x for subset in inds for x in subset]
        realizations = [x for subset in realizations for x in subset]
        realizations = pd.Series(realizations)
        realizations.index = inds
        inds_finite = inds  # [x for subset in inds for x in subset]
        realizations_finite = [x for subset in realizations_finite for x in subset]
        realizations_finite = pd.Series(realizations_finite)
        realizations_finite.index = inds_finite
        realizations_finite_data = pd.DataFrame(realizations_finite)
        realizations_finite_data.columns = ['occ']
        arr_adj_hat, dep_adj_hat, _, _ = estimate_adjusted_rate_sim(realizations_finite_data, 15, silent=True)
        days = np.unique(realizations_finite.index.date)
        d_, pred_ = calc_some_true_and_preds_1H(days, realizations_finite, estimate_adjusted_rate_sim, realizations_finite_data, realizations)
        d_true.append(d_)
        d_pred.append(pred_)
    d_pred = [x for subset in d_pred for x in subset]
    d_true = [x for subset in d_true for x in subset]
    print('Successfully calculated RMSE for {} days:'.format(n_months*29), np.sqrt(mean_squared_error(y_true=d_true, y_pred=d_pred)))
    return np.sqrt(mean_squared_error(y_true=d_true, y_pred=d_pred))