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

API_KEY_GOOGLE = "AIzaSyDl8r2Vup0mrmphy8AtRD8IqV-TgOslb4A"



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
    ax.plot(wd_1)
    ax.plot(we_1)
    ax.set_title('{} station availability'.format(station_1.capitalize().replace('_', ' ')))
    ax.set_ylabel('# of bikes (out of 30)')
    ax.set_xlabel('hour of the day')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.legend(['weekdays', 'weekends'])

    ax1.plot(wd_2)
    ax1.plot(we_2)
    ax1.set_title('{} station availability'.format(station_2.capitalize().replace('_', ' ')))
    ax1.set_ylabel('# of bikes (out of 30)')
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
    #fig = plt.figure(figsize=(12, 6))
    plt.plot(dff['full'])
    plt.plot(dff['empty'])
    plt.legend(labels=['full for 1.5+ hrs', 'empty for 1.5+ hrs'])
    plt.xlabel('Month')
    plt.ylabel('Number of stations')
    #fig.savefig('empty_full.png');
    plt.show()