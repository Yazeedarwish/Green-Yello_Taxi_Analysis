#!/usr/bin/env python
# coding: utf-8
# Test_func.py
import numpy as np
import pandas as pd
import pandas as pd
import numpy as np


from sklearn.preprocessing import LabelEncoder
#from geopy.geocoders import Nominatim
from sklearn import preprocessing
from sqlalchemy import create_engine
import os
import plotly.express as px 
from dash import * 


def establish_connection(db_name):
	
	engine = create_engine(f'postgresql://root:root@pgdatabase:5432/{db_name}')

	engine.connect()
	# return engine(connection cursor)
	return engine

# example func. you might use in the milestone
def check_clean_csv(file_to_check):
	if os.path.exists(file_to_check):
		return True
	return False

def upload_csv(filename,table_name,engine):  
	df = pd.read_csv(filename)
	try:
		df.to_sql(table_name, con = engine, if_exists='fail', index=False)
		print('csv file uploaded to the db as a table')
	except ValueError as e:
		print("Table already exists. Error:", e)

# In[1]:


def clean_columname(taxi_df):
    #convert from string to datetime
    taxi_df['lpep pickup datetime'] = pd.to_datetime(taxi_df['lpep pickup datetime'])
    taxi_df['lpep dropoff datetime'] = pd.to_datetime(taxi_df['lpep dropoff datetime'])
    
    clean_data = taxi_df.copy()
    clean_data.columns = taxi_df.columns.str.replace(' ', '_').str.lower()
    
    
    return clean_data


# In[2]:


def remove_all_duplicates(taxi_df):
    clean_data = taxi_df.copy()
    
    #removing normal duplicates 
    clean_data = clean_data.drop_duplicates()
    
    #removing duplicates that appears as negative
    taxi_copy = clean_data.copy()
    
    num = taxi_copy.select_dtypes(include=['int', 'float'])
    for col in num:
        taxi_copy[col] = taxi_copy[col].abs()
        
    duplicates = taxi_copy.duplicated()
    # Use boolean indexing to get the indices of duplicated rows
    duplicated_indices = taxi_copy.index[duplicates]
    
    clean_data = clean_data.drop(duplicated_indices)
    
    return clean_data
    


# In[3]:


def remove_diff_cases(taxi_df):
    #case 1
    clean_data = taxi_df.copy()
    neg_df = clean_data[clean_data['fare_amount'] < 0]
    ind = neg_df[( ((neg_df['payment_type'] == "Cash") | (neg_df['payment_type'] == "Credit Card") ) & (neg_df['trip_distance'] == 0))].index
    clean_data = clean_data.drop(ind)
    
    #case 2
    
    idx2 = neg_df[( ((neg_df['payment_type'] == "Cash") | (neg_df['payment_type'] == "Credit Card") ) & (neg_df['trip_distance'] != 0))].index
    numeric = clean_data.select_dtypes(include=['int', 'float'])
    for column in numeric:
        for index in idx2:
            if index in clean_data.index:  # Check if the index exists in the DataFrame
                clean_data.at[index, column] = abs(clean_data.at[index, column])
                
    #case 3
    
    No_charge_cond_index = clean_data[((clean_data['payment_type'] == 'No charge') & (clean_data['trip_distance'] == 0) & (clean_data['fare_amount'] < 0)) ].index
    columns_to_zero = ['fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount', 'improvement_surcharge', 'total_amount']

    for col in columns_to_zero:
        for i in No_charge_cond_index:
            clean_data.loc[i,col] = 0
            
    #case 4 
    
    dispute_ind = clean_data[((clean_data['fare_amount'] < 0) & (clean_data['payment_type'] == 'Dispute') & (clean_data['trip_distance'] == 0 ))].index
    clean_data = clean_data.drop(dispute_ind)
    
    #case 5
    
    copy = clean_data.copy()
    
    columns_to_sum = ['fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount', 'improvement_surcharge']
    copy['total_payment'] = copy[columns_to_sum].sum(axis=1) # Sum all except total
    
    diff = copy['total_payment'] - copy['total_amount']
    con_diff = diff > 0
    
    copy.loc[con_diff, 'total_amount'] = copy.loc[con_diff,'total_payment']
    clean_data = copy.drop('total_payment', axis=1)
    
    #drop because complete missing
    clean_data = clean_data.drop(['ehail_fee', 'congestion_surcharge' ], axis=1)
    
    return clean_data
    


# In[4]:


def add_columns(taxi_df):
    clean_data = taxi_df.copy()
    clean_data['trip_duration'] = clean_data['lpep_dropoff_datetime'] - clean_data['lpep_pickup_datetime']
    clean_data['trip_duration_hours'] = clean_data['trip_duration'].dt.total_seconds() / 3600  # Duration in hours
    
    return clean_data


# In[5]:


def handle_missing(taxi_df):
    clean_data = taxi_df.copy()
    
    #passenger_count column------------------------------------------
    clean_data['passenger_count'].fillna(clean_data['passenger_count'].median(), inplace=True)
    #print("passengercount : ",clean_data['passenger_count'].isnull().sum() )
    
    #extra column---------------------------------
    pu_hours = clean_data['lpep_pickup_datetime'].dt.hour
    
    condition_one = ((clean_data['extra'].isnull()) & ((pu_hours == 16) | (pu_hours == 17) | (pu_hours == 18) | (pu_hours == 19)))
    # Use .loc to assign the value to the 'extra' column for rows that meet the condition
    clean_data.loc[condition_one, 'extra'] = 1
    
    condition_half = ((clean_data['extra'].isnull()) & ((pu_hours == 20) | (pu_hours == 21) | (pu_hours == 22) | (pu_hours == 23)))
    clean_data.loc[condition_half, 'extra'] = 0.5
    
    condition_else = (clean_data['extra'].isnull())
    clean_data.loc[condition_else, 'extra'] = clean_data.loc[condition_else, 'total_amount'] - clean_data.loc[condition_else, 'improvement_surcharge'] - clean_data.loc[condition_else, 'tolls_amount'] - clean_data.loc[condition_else, 'tip_amount']
   # print("extra : ", clean_data['extra'].isnull().sum())
    
    # payment type --------------------------------------------------------
    
    clean_data['payment_type'].fillna(value=clean_data['payment_type'].mode()[0], inplace=True)
    #print("payment_type", clean_data['payment_type'].isnull().sum())
    
    
    return clean_data
    
    
    
    


# In[6]:


def get_cutoff(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    cut_off = IQR * 2.5 
    lower = Q1 - cut_off
    upper =  Q3 + cut_off
    return lower,upper


# In[7]:


def floor_and_cap(column):
    floor=column.quantile(0.1)
    cap=column.quantile(0.9)
    return floor,cap


# In[8]:


def handle_outliers(taxi_df):
    clean_data = taxi_df.copy()
    
    # passenger_count outliers--------------------------------------
    clean_data = clean_data.drop(clean_data[clean_data['passenger_count'] > 6].index)
     
    
    #Trip_distance outliers ----------------------------------------
    lower, upper=get_cutoff(clean_data.trip_distance)
    df1_trip = clean_data[clean_data.trip_distance < lower]
    df2_trip = clean_data[clean_data.trip_distance > upper]
    
    floor, cap=floor_and_cap(clean_data.trip_distance)
    for i in df1_trip.index:
        clean_data.at[i, 'trip_distance'] = floor

    for i in df2_trip.index:
        clean_data.at[i, 'trip_distance'] = cap
        
      
    #fare_amount outliers -----------------------------------------
    clean_data = clean_data.drop(clean_data[clean_data.fare_amount > 200].index)
    
    clean_data['trip_duration_minutes'] = (clean_data['trip_duration'].dt.total_seconds() % 3600) / 60
    c = clean_data[((clean_data.fare_amount>0) & (clean_data.trip_distance==0) & ((clean_data.trip_duration_hours == 0) & (clean_data.trip_duration_minutes == 0) ))].index
    clean_data = clean_data.drop(c) 
    
    lower, upper=get_cutoff(clean_data.fare_amount)
    df1_fare = clean_data[clean_data.fare_amount < lower]
    df2_fare= clean_data[clean_data.fare_amount > upper]
    
    floor, cap=floor_and_cap(clean_data.fare_amount)
    for i in df1_fare.index:
        clean_data.at[i, 'fare_amount'] = floor

    for i in df2_fare.index:
        clean_data.at[i, 'fare_amount'] = cap
        
    
        
    
    #Extra outliers ----------------------------------------------------
    
    lower, upper=get_cutoff(clean_data.extra)
    df1_extra = clean_data[clean_data.extra > upper]
    df2_extra = clean_data[clean_data.extra < lower]
    
    floor, cap = floor_and_cap(clean_data.extra)

    for i in df1_extra.index:
        clean_data.at[i, 'extra'] = floor

    for i in df2_extra.index:
        clean_data.at[i, 'extra'] = cap
    
    
        
        
    #Tip_amount outliers-----------------------------
    lower, upper=get_cutoff(clean_data.tip_amount)
    df1_tip = clean_data[clean_data.tip_amount < lower]
    df2_tip = clean_data[clean_data.tip_amount > upper]
    
    floor, cap=floor_and_cap(clean_data.tip_amount)

    for i in df1_tip.index:
        clean_data.at[i, 'tip_amount'] = floor

    for i in df2_tip.index:
        clean_data.at[i, 'tip_amount'] = cap
        
    
        
    #Tolls_amount outliers------------------------------
    lower, upper=get_cutoff(clean_data.tolls_amount)
    df1_tolls = clean_data[clean_data.tolls_amount < lower]
    df2_tolls = clean_data[clean_data.tolls_amount > upper]
    
    clean_data = clean_data.drop(clean_data[clean_data.tolls_amount > 50].index)
    
    clean_data.loc[clean_data['tolls_amount'] > 10, 'tolls_amount'] = clean_data['tolls_amount'].median()
    
    
    
    #improvement_surcharge -----------------------------------------------
    clean_data = clean_data.drop(clean_data[clean_data.improvement_surcharge == -0.3].index)
    
    
    #Total_amount -------------------------------------------------------------
    columns_to_sum = ['fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount', 'improvement_surcharge']
    clean_data['total_amount'] = clean_data[columns_to_sum].sum(axis=1)
   
    
    
    
    return clean_data
    


# In[9]:


def discretize_data_into_weeks(dataframe, pickup_column):
    # Make a copy of the input DataFrame to avoid modifying the original
    df = dataframe.copy()

    # Convert date columns to datetime objects
    df[pickup_column] = pd.to_datetime(df[pickup_column])

    # Calculate the week numbers for pickup and dropoff dates
    df['week_number'] = df[pickup_column].dt.strftime('%U').astype(int)
 

    # Create the 'Date range' based on the start and end dates of each week (7 days per week)
    df['date_range'] = df[pickup_column].dt.strftime('%Y-%m-%d') + ' to ' + (df[pickup_column] + pd.DateOffset(days=6)).dt.strftime('%Y-%m-%d')

    return df


# In[10]:


def encoding(taxi_df):
    clean_data = taxi_df.copy()
    
    clean_data = pd.get_dummies(clean_data, columns=['vendor', 'trip_type', 'payment_type', 'rate_type', 'store_and_fwd_flag'])
    
    label_encoder = LabelEncoder()
    clean_data['pickup_loc_encoded'] = label_encoder.fit_transform(clean_data['pu_location'])
    clean_data['dropoff_loc_encoded'] = label_encoder.fit_transform(clean_data['do_location'])
    
    return clean_data
    


# In[11]:


def is_weekend(date_range):
    # Split the date range to extract start and end dates
    start_date, end_date = date_range.split(' to ')

    # Convert start_date and end_date to datetime objects
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Check if either the start_date or end_date falls on a weekend
    if (start_date.dayofweek >= 5) or (end_date.dayofweek >= 5):
        return 'Weekend'
    else:
        return 'Normal day'


# In[12]:


def add_last_feature(taxi_df):
    clean_data = taxi_df.copy()
    # Apply the is_weekend function to the 'date_range' column and create a new column 'day_type'
    clean_data['day_type'] = clean_data['date_range'].apply(is_weekend)
    
    return clean_data



def clean_dataset(taxi_df):
    after_naming = clean_columname(taxi_df)
    
    after_duplicates = remove_all_duplicates(after_naming )
    
    after_cases = remove_diff_cases(after_duplicates)
    
    after_adding_col = add_columns(after_cases)
    
    after_missing = handle_missing(after_adding_col)
    
    after_outliers = handle_outliers(after_missing)
    
    after_discretize = discretize_data_into_weeks(after_outliers, 'lpep_pickup_datetime')
    
    after_encoding = encoding(after_discretize)
    
    after_add_feature = add_last_feature(after_encoding)
    
    clean_data = after_add_feature
    
    return clean_data



def filling_lookup(df, col, lookup_table):
    encoded_values=preprocessing.LabelEncoder().fit_transform(df[col])
    map_values=pd.DataFrame({
    'Column Name' : [col] * len(encoded_values),
    'Original Value': df[col],
    'Imputed Value' : encoded_values})

    return pd.concat([lookup_table, map_values.drop_duplicates()])


def look_up(df):
     lookup_table = pd.DataFrame()
     lookup_table=filling_lookup(df, 'pu_location', lookup_table)
     lookup_table=filling_lookup(df, 'do_location', lookup_table)

     return lookup_table


#Dashboard creation

def create_trip_distance_by_hour_plot(taxi_df):
    # Convert datetime columns to datetime objects
    taxi_df['lpep_pickup_datetime'] = pd.to_datetime(taxi_df['lpep_pickup_datetime'])

    # Create a DataFrame with 'hours' and 'trip distance' columns
    hourly_data = taxi_df[['lpep_pickup_datetime', 'trip_distance']]
    hourly_data['hours'] = hourly_data['lpep_pickup_datetime'].dt.hour

    # Group by hours and calculate mean trip distance
    mean_trip_distance_by_hour = hourly_data.groupby('hours')['trip_distance'].mean().reset_index()

    # Create a line plot using Plotly Express
    fig = px.line(mean_trip_distance_by_hour, x='hours', y='trip_distance',
                  labels={'trip_distance': 'Mean Trip Distance', 'hours': 'Hour of the Day'},
                  title="Mean Trip Distance by Hour of the Day")
    
    return fig


def create_mean_tip_amount_by_location_plot(taxi_df):
    # Calculate mean tip amount by pickup location
    ress = taxi_df.groupby('pu_location')['tip_amount'].mean()
    first_ress = ress.sort_values(ascending=False).head(30)

    # Create a bar plot using Plotly Express
    fig = px.bar(
        x=first_ress.index,
        y=first_ress.values,
        labels={'x': 'Pickup Location', 'y': 'Mean Tip Amount'},
        title="Mean Tip Amount by Pickup Location",
    )
    return fig



def create_tolls_amount_by_do_location_plot(taxi_df):
    # Calculate mean tolls amount by DO Location
    mean_tolls_amount = taxi_df.groupby('do_location')['tolls_amount'].mean().sort_values(ascending=False)[0:15]

    # Create a bar plot using Plotly Express
    fig = px.bar(
        x=mean_tolls_amount.index,
        y=mean_tolls_amount.values,
        labels={'x': 'DO Location', 'y': 'Mean Tolls Amount'},
        title='Mean Tolls Amount by DO Location',
    )
    return fig



