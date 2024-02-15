# Green-Yello_Taxi_Analysis
## Project Description:
This project involves the analysis and preprocessing of New York City taxi datasets, specifically focusing on the green taxi dataset and the yellow taxi dataset. Through a series of milestones, the project aims to explore the data, engineer relevant features, preprocess it for various analytical tasks, and orchestrate the workflow using Docker and Apache Airflow. The technologies employed include Python libraries such as Pandas and PySpark, along with PostgreSQL for data warehousing. SQL queries are utilized post data loading into PostgreSQL for further analysis and data manipulation. The ultimate goal is to gain insights into transportation patterns and facilitate data-driven decision-making processes.

# Milestones:

## 1- Milestone 1:
Involves loading the green taxi dataset, performing exploratory data analysis with visualization, extracting additional data, feature engineering, and preprocessing the data for downstream tasks such as machine learning and further analysis.

## 2- Milestone 2:
Requires packaging the code from Milestone 1 into a Docker image and setting up a PostgreSQL database for storing cleaned datasets and lookup tables. After loading the data into PostgreSQL, SQL queries are utilized for further analysis and data manipulation.

## 3- Milestone 3:
Involves preprocessing the New York yellow taxi dataset using PySpark. The same month and year used for the green taxis in Milestone 1 are utilized for consistency. Basic data preparation and analysis are performed to gain insights into the data.

## 4- Milestone 4:
Orchestrates the tasks from Milestones 1 and 2 using Apache Airflow within Docker. The green dataset is primarily processed using pandas for simplicity. Tasks include reading the CSV file, cleaning and transforming the data, loading it into PostgreSQL, extracting additional resources such as GPS coordinates, integrating with the cleaned dataset, and creating a dashboard using the Dash package in Python for web interface presentation.

# Technologies Used:
Python (including libraries such as Pandas, PySpark, Dash)
PostgreSQL
Docker
Apache Airflow

## Purpose:
This project serves as a practical exploration of data engineering techniques applied to transportation datasets, demonstrating proficiency in data preprocessing, workflow orchestration, SQL querying, and dashboard creation for data-driven insights.
