import xarray as xr
import numpy as np
import pandas as pd

z500 = xr.open_mfdataset('./data/2m_temperature_nc/*.nc', combine='by_coords')

# array = z500.t2m.values
# np.save('./data/2m_temperature/total.npy', array)

data = []

for i in range(0, len(z500.time)):
    datetime_value = z500.time.values[i]

    year = pd.to_datetime(datetime_value).year
    month = pd.to_datetime(datetime_value).month
    day = pd.to_datetime(datetime_value).day
    time = pd.to_datetime(datetime_value).time()
    
    data.append({'Year': year, 'Month': month, 'Day': day, 'Time': time,  'Timestep': i})

df = pd.DataFrame(data, columns=['Year', 'Month', 'Day', 'Time', 'Timestep'])
df.to_csv('./data/2m_temperature/total.csv', index=False)