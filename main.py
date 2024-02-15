from doctest import debug
from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

import pandas as pd
import numpy as np
from functions import *

# For Label Encoding
from sklearn import preprocessing
import dash
import dash_core_components as dcc
import dash_html_components as html
from sqlalchemy import create_engine

dataset = 'green_tripdata_2016-11'

#Task1 
def extract_clean(filename):
    
    # Check if the clean CSV file already exists
    if (check_clean_csv('/opt/airflow/data/green_taxi_clean.csv') == False):
        # If not, perform the cleaning
        df = pd.read_csv(filename)
        cleaned = clean_dataset(df)
        
        # Export the cleaned DataFrame to a CSV file
        cleaned.to_csv('/opt/airflow/data/green_taxi_clean.csv', index=False)

    df = pd.read_csv('/opt/airflow/data/green_taxi_clean.csv')

    lookup_table = look_up(df)
    lookup_table.to_csv('/opt/airflow/data/lookup_table.csv', index=False)
    print('loaded after cleaning succesfully')




#Task 2
def add_coordinates(filename,coordinates_file):
    df = pd.read_csv(filename)

    
    df_coordinates = pd.read_csv(coordinates_file)

    data_dict = dict(zip(df_coordinates['Location'], zip(df_coordinates['Latitude'], df_coordinates['Longitude'])))
    df['pu_coordinates'] = df['pu_location'].map(data_dict)
    df['do_coordinates'] = df['do_location'].map(data_dict)
    df.to_csv('/opt/airflow/data/green_taxi_with_coordinates.csv',index=False)



#Task 3    
def load_to_postgres(filename,filename_lookup): 
    df = pd.read_csv(filename)
    lookup_table = pd.read_csv(filename_lookup)

    engine = create_engine('postgresql://root:root@pgdatabase:5432/green_taxi_etl')
    if(engine.connect()):
        print('connected succesfully')
    else:
        print('failed to connect')
    df.to_sql(name = 'M4_green_taxis_11_2016',con = engine,if_exists='replace')
    lookup_table.to_sql(name = 'lookup_table',con = engine,if_exists='replace')



def create_dashboard(filename):
    df = pd.read_csv(filename)
     # Initialize the Dash app
    app = dash.Dash()

    layout1 = html.Div([
        html.H1("Yazeed Ahmed 49-11916 NETW", style={'text-align': 'center'}),
        html.Br(),
        html.H1("Green taxi dataset", style={'text-align': 'center'}),
        html.Br(),
        html.Div(),
        html.H1("Mean Tip Amount by Pickup Location", style={'text-align': 'center'}),
        dcc.Graph(figure=create_mean_tip_amount_by_location_plot(df),
                style={'height': '800px'})  # Adjust the height as needed
    ])

    # Create layouts for each graph using functions
    layout2 = html.Div([
        html.H1("Yazeed Ahmed 49-11916 NETW", style={'text-align': 'center'}),
        html.Br(),
        html.H1("Green taxi dataset", style={'text-align': 'center'}),
        html.Br(),
        html.Div(),
        html.H1("Relationship between hours of day and mean trip distance", style={'text-align': 'center'}),
        dcc.Graph(figure=create_trip_distance_by_hour_plot(df))
    ])



    layout3 = html.Div([
        html.H1("Yazeed Ahmed 49-11916 NETW", style={'text-align': 'center'}),
        html.Br(),

        html.H1("Mean Tolls Amount by DO Location", style={'text-align': 'center'}),
        dcc.Graph(figure=create_tolls_amount_by_do_location_plot(df),
                style={'height': '800px'})
    ])

    # Combine layouts into one app layout
    app.layout = html.Div([
        dcc.Tabs([
            dcc.Tab(label='Show graph 1', children=[layout1]),
            dcc.Tab(label='Show graph 2', children=[layout2]),
            dcc.Tab(label='Show graph 3', children=[layout3]),
        ])
    ])
    app.run_server(host='0.0.0.0', debug= False)
    print('dashboard is successful and running on port 8000')






default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    'start_date': days_ago(2),
    "retries": 1,
}

dag = DAG(
    'green_taxi_etl_pipeline',
    default_args=default_args,
    description='green taxi etl pipeline',
)
with DAG(
    dag_id = 'green_taxi_pipeline',
    schedule_interval = '@once',
    default_args = default_args,
    tags = ['green_taxi-pipeline'],
)as dag:
    extract_clean_task= PythonOperator(
        task_id = 'extract_dataset',
        python_callable = extract_clean,
        op_kwargs={
            "filename": '/opt/airflow/data/green_tripdata_2016-11.csv'
        },
    )
    add_coordinates_task= PythonOperator(
        task_id = 'add_gps_coordinates',
        python_callable = add_coordinates,
        op_kwargs={
            "filename": "/opt/airflow/data/green_taxi_clean.csv",
            "coordinates_file": "/opt/airflow/data/unique_locations_with_gps.csv"
        },
    )
    load_to_postgres_task=PythonOperator(
        task_id = 'load_to_postgres',
        python_callable = load_to_postgres,
        op_kwargs={
            "filename": "/opt/airflow/data/green_taxi_with_coordinates.csv",
            "filename_lookup": "/opt/airflow/data/lookup_table.csv"
        },
    )
    create_dashboard_task= PythonOperator(
        task_id = 'create_dashboard_task',
        python_callable = create_dashboard,
        op_kwargs={
            "filename": "/opt/airflow/data/green_taxi_with_coordinates.csv"
        },
    )
    


    extract_clean_task >> add_coordinates_task >> load_to_postgres_task >> create_dashboard_task


