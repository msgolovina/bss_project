import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import branca
import folium
import urllib
import json

from sklearn.cluster import KMeans

API_KEY_GOOGLE = "AIzaSyDl8r2Vup0mrmphy8AtRD8IqV-TgOslb4A"


def altitude(latitude, longitude):
    '''
    this function takes points indicating station locations and
    requests elevation of these points from Google Maps API Elevation Service
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

    figure, (axis_weekdays, axis_weekends) = plt.subplots(1, 2, figsize=(20, 6))

    sses = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(weekdays_norm.T)
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


def plot_cluster_usage_patterns(weekdays_norm, weekends_norm, cluster_colours):
    figure, (axis_weekdays, axis_weekends) = plt.subplots(1, 2, figsize=(20, 6))

    # weekdays
    n_clusters_weekdays = 3
    model = KMeans(n_clusters=n_clusters_weekdays, random_state=7)
    model_fit = model.fit(weekdays_norm.T)
    for k, colour in zip(model_fit.cluster_centers_, cluster_colours):
        axis_weekdays.plot(100 * k, color=colour, label=colour)
    axis_weekdays.set_ylim([26, 100])
    xticks = range(0, 289, 24)
    xticks_labels = ["%02d:%02d" % divmod(i * 5, 60) for i in [int(tick) for tick in xticks]]
    axis_weekdays.set_xticks(xticks)
    axis_weekdays.set_xticklabels(xticks_labels)
    axis_weekdays.set_xlabel('hour of the day')
    axis_weekdays.set_ylabel('% of the station full')
    axis_weekdays.set_title('Clusters of bike usage patterns at weekdays')
    axis_weekdays.margins(0.02)
    axis_weekdays.legend(labels=['balanced', 'full at working time', 'empty at working time'])
    clusters_weekdays = model_fit.predict(weekdays_norm.T)

    # weekends

    n_clusters_weekends = 4
    model = KMeans(n_clusters=n_clusters_weekends, random_state=7)
    model_fit = model.fit(weekends_norm.T)
    for k, colour in zip(model_fit.cluster_centers_, cluster_colours):
        axis_weekends.plot(100 * k, color=colour, label=colour)
    axis_weekends.set_ylim([55, 100])
    xticks = range(0, 289, 24)
    xticks_labels = ["%02d:%02d" % divmod(i * 5, 60) for i in [int(tick) for tick in xticks]]
    axis_weekends.set_xticks(xticks)
    axis_weekends.set_xticklabels(xticks_labels)
    axis_weekends.set_xlabel('hour of the day')
    axis_weekends.set_ylabel('% of the station full')
    axis_weekends.set_title('Clusters of bike usage patterns at weekends')
    axis_weekends.margins(0.02)
    axis_weekends.legend(
        labels=['emptying with time', 'full at working time', 'refilling with time', 'empty at working time'])
    clusters_weekends = model_fit.predict(weekends_norm.T)

    return clusters_weekdays, clusters_weekends