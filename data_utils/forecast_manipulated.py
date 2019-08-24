import os
import json
import time, datetime
import numpy as np
from netCDF4 import Dataset

import config as config

root_dir = os.path.join(config.storage_dir, 'forecast')

def get_120ha_date(date):
    ntime = datetime.datetime.strptime(date, '%Y%m%d') + datetime.timedelta(hours=120)
    return ntime.strftime('%Y%m%d')

def create_forecast_data_glob():
    with open(os.path.join(config.forecast_raw_dir, 'fc_idx')) as f:
        j = json.load(f)

    all_data = []
    all_times = []
    all_time_paths = []
    all_dates = list(j)
    date_files = os.listdir(os.path.join(config.forecast_raw_dir, 'temperature2m'))
    for date in all_dates:
        for file in [f'MD_ECMF_PD_temperature2m_IT_{date}{t:02}_FH_120_VT_{get_120ha_date(date)}{t:02}.nc' for t in [0, 12]]:
            if file in date_files:
                data = Dataset(os.path.join(config.forecast_raw_dir, 'temperature2m', file))
                raw_data = np.array(data.variables['t2m']).squeeze().transpose((2 ,1 ,0))[:141]
                all_data.append(raw_data)
                all_time_paths += os.path.join(config.forecast_raw_dir, 'temperature2m', file)
                all_times.append(time.mktime(time.strptime(file.split('_')[9][:-3], '%Y%m%d%H')))
            if all_times.__len__() % 100 == 0:
                print(all_times.__len__())


    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    rootgrp = Dataset(os.path.join(root_dir, 'data.nc'), 'w')

    rootgrp.createDimension('time', all_times.__len__())
    rootgrp.createDimension('latitude', config.latitude.__len__())
    rootgrp.createDimension('longitude', config.longitude.__len__())
    rootgrp.createDimension('ens', 51)

    times = rootgrp.createVariable('time', 'u4', ('time',))
    times.units = 'Second since 1970-01-01T08:00:00Z'
    latitudes = rootgrp.createVariable('latitude', 'f4', ('latitude',))
    longitudes = rootgrp.createVariable('longitude', 'f4', ('longitude',))
    data = rootgrp.createVariable('data', 'f4', ('time', 'longitude', 'latitude', 'ens'))

    times[:] = np.array(all_times)
    latitudes[:] = np.array(config.latitude)
    longitudes[:] = np.array(config.longitude)
    data[:,:,:,:] = np.array(all_data)[:,1:,1:,:]

    rootgrp.close()

create_forecast_data_glob()
