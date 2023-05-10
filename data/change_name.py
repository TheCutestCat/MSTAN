import pandas as pd
import os
def change_data_name(new_name):
    columns = {'Time(year-month-day h:m:s)' : 'time',
               'Wind speed at height of 10 meters (m/s)' : 'wind10',
               'Wind direction at height of 10 meters (˚)' : 'angle10',
               'Wind speed at height of 30 meters (m/s)' : 'wind30',
               'Wind direction at height of 30 meters (˚)' : 'angle30',
               'Wind speed at height of 50 meters (m/s)' : 'wind50',
               'Wind direction at height of 50 meters (˚)' : 'angle50',
               'Air temperature  (°C) ' : 'temp',
               'Atmosphere (hpa)' : 'atmosphere',
               'Relative humidity (%)' : 'humidity',
               'Power (MW)' : 'power'
           }
    drop_list = ['Wind speed - at the height of wheel hub (m/s)', 'Wind speed - at the height of wheel hub (˚)']

    path = os.path.join('../data',new_name)
    df = pd.read_excel(path)
    df = df.rename(columns=columns, inplace=True)
    df.drop(drop_list, axis=1, inplace=True)
    df.to_csv('../data/data_changed_name.csv')
    print(f'name changed and dat saved to {path}')

change_data_name('Wind farm site 1 (Nominal capacity-99MW).xlsx')