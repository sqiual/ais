import pandas as pd
import numpy as np
import os


min_voy = 20  # minimum voyage duration in minutes
min_speed = 1
re_sample_intvl = 10
tim_interval = np.arange(0, 3600*24, 10)
min_ves = 1
dataset_name = 'test'
dataset_path = dataset_name+'.csv'


dr = os.path.join('stgcnn/data', dataset_path)
df = pd.read_csv(dr)
raw_total_voy = len(df['MMSI'].unique())

df[['date', 'time']] = df['# Timestamp'].str.split(' ', 1, expand = True)
data = df[['date','time','MMSI','Latitude','Longitude','SOG','COG']]
data = data[data['MMSI'].map(lambda x: len(str(x)) == 9)]
data = data.loc[data['SOG'].notnull()]

data['time'] = pd.to_timedelta(data['time']).dt.total_seconds()
# data['rel_lat'] = data.groupby(['MMSI'])['Latitude'].diff().fillna(0)
# data['rel_long'] = data.groupby(['MMSI'])['Longitude'].diff().fillna(0)
# data['interval'] = data.groupby(['MMSI'])['time'].diff().fillna(0)

data = data.groupby(['MMSI']).filter(lambda x: x['SOG'].mean() > min_speed)
data = data.groupby(['MMSI']).filter(lambda x: (x['time'].max() - x['time'].min())/60 > min_voy)

clean_voy = len(data['MMSI'].unique())
print("data exclude mmsi !=9 & mean(SOG) < 1 & voyage < 30 min: %f %%" %(100*clean_voy/raw_total_voy))
print('usable voyage:', clean_voy)

# interpolation begins
#==============================================================
#==============================================================
# data.index = data.index.droplevel()
data = data.groupby('MMSI')

# start and end time of each voyage
group_max = round(data['time'].max()/10)*10
group_min = round(data['time'].min()/10)*10
new_data = pd.merge(group_min, group_max,left_index = True, right_index = True)
new_data['MMSI'] = new_data.index

resam_time = pd.Series([np.arange(time_x, time_y, re_sample_intvl) for time_x, time_y in zip(new_data.time_x, new_data.time_y)], new_data.index)

cleaned = np.empty([1,6])
for i in range(resam_time.shape[0]):
    re_time = resam_time.values[i]
    mmsi = np.repeat(resam_time.index[i], re_time.shape[0])
    org_time = data.get_group(resam_time.index[i])['time'].to_numpy()
    #
    org_lat = data.get_group(resam_time.index[i])['Latitude'].to_numpy()
    org_long = data.get_group(resam_time.index[i])['Longitude'].to_numpy()
    org_sog = data.get_group(resam_time.index[i])['SOG'].to_numpy()
    org_cog = data.get_group(resam_time.index[i])['COG'].to_numpy()
    #
    re_lat = np.interp(x = re_time, xp = org_time, fp = org_lat)
    re_long = np.interp(x=re_time, xp=org_time, fp=org_long)
    re_sog = np.interp(x=re_time, xp=org_time, fp=org_sog)
    re_cog = np.interp(x=re_time, xp=org_time, fp=org_cog)
    #
    clean = np.stack((re_time, mmsi, re_lat,re_long,re_sog,re_cog),axis = 1)
    cleaned = np.concatenate((cleaned, clean),axis = 0)
    # print(clean.shape) [xxx, 6] [time,mmsi,lat,long,sog,cog]
cleaned = cleaned[1:,0:4]
np.savetxt(dataset_name+'.txt',cleaned, delimiter=' ')