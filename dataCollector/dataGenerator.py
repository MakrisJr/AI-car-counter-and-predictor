# how to run: python dataGenerator.py or python3 dataGenerator.py


import pandas as pd
import numpy as np
from sklearn.utils import resample
import datetime

# Get the current month and year
current_date = datetime.datetime.now()
current_month = current_date.month
current_year = current_date.year

filename = f"traffic_data_{current_month}_{current_year}.csv"

# Load data from CSV
df = pd.read_csv(filename)

# Convert the '5 Minutes' column to a datetime format if it's not already
df['5 Minutes'] = pd.to_datetime(df['5 Minutes'])

# Create new features from the datetime column
df['year'] = df['5 Minutes'].dt.year
df['month'] = df['5 Minutes'].dt.month
df['day'] = df['5 Minutes'].dt.day
df['hour'] = df['5 Minutes'].dt.hour
df['minute'] = df['5 Minutes'].dt.minute

## Get min and max 'Lane 1 Flow' for each time
flow_range = df.groupby(df['5 Minutes'].dt.time)['Lane 1 Flow (Veh/5 Minutes)'].agg(['min', 'max'])

# Resample the DataFrame
df_resampled = resample(df, replace=True, n_samples=1000, random_state=42)

# Generate a new 'Lane 1 Flow' for each row in the resampled DataFrame
df_resampled['Lane 1 Flow (Veh/5 Minutes)'] = [np.random.randint(flow_range.loc[t]['min'], flow_range.loc[t]['max'] + 1) for t in df_resampled['5 Minutes'].dt.time]

# Convert the year, month, day, hour, and minute columns back to datetime
df_resampled['5 Minutes'] = pd.to_datetime(df_resampled[['year', 'month', 'day', 'hour', 'minute']])

# Drop the year, month, day, hour, and minute columns
df_resampled = df_resampled.drop(['year', 'month', 'day', 'hour', 'minute'], axis=1)

# Reorder the columns to match the original data
df_resampled = df_resampled[['5 Minutes', 'Lane 1 Flow (Veh/5 Minutes)', '# Lane Points', '% Observed']]

# Save the resampled DataFrame to a CSV
df_resampled.to_csv('resampled_data.csv', index=False)
