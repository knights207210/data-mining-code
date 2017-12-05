import pandas as pd
import numpy as np
import psycopg2
from math import radians, cos, sin, asin, sqrt

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# output path
output_path = "C:/Users/yawen/Google Drive/spatial temporal data analytics/urban computing/TakeoutAnalysis/hubTransitionAnalysis/"

# connect to database and get cursor
try:
    conn = psycopg2.connect(database = 'urbandata', user = 'postgres', host = 'localhost', port = '5432', password = '1234')
except:
    print("I am unable to connect to the database")
cur = conn.cursor()

def userFiltering():
    """
    Fiter users by 1) 2 hubs distance, 2) hub count + period, 3) temporal range, 4) jhr not NULL
    """

    cur.execute('select pass_uid, hub_id, count_sum, count_weekday, count_weekend, weekday_mor, weekday_lun, weekday_tea, weekday_din, weekday_nig, weekend_mor, weekend_lun, weekend_tea, weekend_din, weekend_nig, max, min, center_lat, center_lon, jhr from user_hub_temporal_feature_max_min_center_jhr')
    user_two_hub_features = pd.DataFrame(cur.fetchall(), columns = ['pass_uid', 'hub_id', 'count_sum', 'count_weekday', 'count_weekend', 'weekday_mor', 'weekday_lun', 'weekday_tea', 'weekday_din', 'weekday_nig', 'weekend_mor', 'weekend_lun', 'weekend_tea', 'weekend_din', 'weekend_nig', 'max', 'min', 'center_lat', 'center_lon', 'jhr'])