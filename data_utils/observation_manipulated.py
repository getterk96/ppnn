import os
import time
from datetime import datetime
import numpy as np
from netCDF4 import Dataset, num2date

import config as config

root_dir = os.path.join(config.storage_dir, 'observation')

def create_observation_data_glob():
    all_data = []
    all_times = []
    all_time_paths = []
    all_dates = os.listdir(config.observation_raw_dir)
    for date in all_dates:
        date_files = os.listdir(os.path.join(config.observation_raw_dir, date))
        for t in [f'{date}{t:02}.000.nc' for t in [0, 12]]:
            if t in date_files:
                data = Dataset(os.path.join(config.observation_raw_dir, date, t))
                raw_data = np.array(data.variables['data']).squeeze()[::10,:1101:10]
                all_data.append(raw_data)
                all_time_paths.append(os.path.join(config.observation_raw_dir, date, t))
                all_times.append(time.mktime(time.strptime('20' + t[:8], '%Y%m%d%H')))
            if all_times.__len__() % 100 == 0:
                print(all_times.__len__())


    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    rootgrp = Dataset(os.path.join(root_dir, 'data.nc'), 'w')

    rootgrp.createDimension('time', all_times.__len__())
    rootgrp.createDimension('latitude', config.latitude.__len__())
    rootgrp.createDimension('longitude', config.longitude.__len__())

    times = rootgrp.createVariable('time', 'u4', ('time',))
    times.units = 'Second since 1970-01-01T08:00:00Z'
    latitudes = rootgrp.createVariable('latitude', 'f4', ('latitude',))
    longitudes = rootgrp.createVariable('longitude', 'f4', ('longitude',))
    data = rootgrp.createVariable('data', 'f4', ('time', 'longitude', 'latitude'))

    times[:] = np.array(all_times)
    latitudes[:] = np.array(config.latitude)
    longitudes[:] = np.array(config.longitude)
    data[:,:,:] = np.array(all_data)[:,1:,1:]

    rootgrp.close()

create_observation_data_glob()
