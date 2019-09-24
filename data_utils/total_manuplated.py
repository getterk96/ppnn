from netCDF4 import Dataset
from os import path
import numpy as np
import config as config

data_dir = config.storage_dir

def get_time_consistant():
    forecast_dir = path.join(data_dir, 'forecast')
    observation_dir = path.join(data_dir, 'observation')
    fc_rg = Dataset(path.join(forecast_dir, 'raw_data.nc'))
    obs_rg = Dataset(path.join(observation_dir, 'raw_data.nc'))
    fc_dates = fc_rg.variables['time']
    obs_dates = obs_rg.variables['time']
    dates = set(fc_dates) & set(obs_dates)
    fc_data = fc_rg.variables['data'][:]
    obs_data = obs_rg.variables['data'][:]
    new_fc_data = []
    new_obs_data = []
    for idx, i in enumerate(fc_dates):
        if fc_dates[idx] in dates:
            new_fc_data.append(fc_data[idx])
    for idx, i in enumerate(obs_dates):
        if obs_dates[idx] in dates:
            new_obs_data.append(obs_data[idx])
    dates = sorted(list(dates))

    rootgrp = Dataset(path.join(forecast_dir, 'data.nc'), 'w')
    rootgrp.createDimension('time', dates.__len__())
    rootgrp.createDimension('latitude', config.latitude.__len__())
    rootgrp.createDimension('longitude', config.longitude.__len__())
    rootgrp.createDimension('ens', 51)

    times = rootgrp.createVariable('time', 'u4', ('time',))
    times.units = 'Second since 1970-01-01T08:00:00Z'
    latitudes = rootgrp.createVariable('latitude', 'f4', ('latitude',))
    longitudes = rootgrp.createVariable('longitude', 'f4', ('longitude',))
    data = rootgrp.createVariable('data', 'f4', ('time', 'longitude', 'latitude', 'ens'))

    times[:] = np.array(dates)
    latitudes[:] = np.array(config.latitude)
    longitudes[:] = np.array(config.longitude)
    data[:,:,:,:] = np.array(new_fc_data)[:,:,:,:]
    rootgrp.close()

    rootgrp = Dataset(path.join(observation_dir, 'data.nc'), 'w')

    rootgrp.createDimension('time', dates.__len__())
    rootgrp.createDimension('latitude', config.latitude.__len__())
    rootgrp.createDimension('longitude', config.longitude.__len__())

    times = rootgrp.createVariable('time', 'u4', ('time',))
    times.units = 'Second since 1970-01-01T08:00:00Z'
    latitudes = rootgrp.createVariable('latitude', 'f4', ('latitude',))
    longitudes = rootgrp.createVariable('longitude', 'f4', ('longitude',))
    data = rootgrp.createVariable('data', 'f4', ('time', 'longitude', 'latitude'))

    times[:] = np.array(dates)
    latitudes[:] = np.array(config.latitude)
    longitudes[:] = np.array(config.longitude)
    data[:,:,:] = np.array(new_obs_data)[:,:,:]
    rootgrp.close()

get_time_consistant()
