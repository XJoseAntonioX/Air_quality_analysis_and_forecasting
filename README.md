# Air Quality Prediction in Monterrey

This repository contains the implementation and analysis of a predictive model aimed at addressing a critical environmental challenge: air pollution in Monterrey, Mexico. Using advanced data science techniques and historical data from January 2022 to July 2024, this project focuses on forecasting key air pollutants (PM10, SO2, CO, and NOX) to support informed decision-making and mitigate the adverse health effects associated with poor air quality.

## Project Objective

The goal of this project is to develop a reliable predictive model capable of estimating the concentrations of major air pollutants in Monterrey. By leveraging a combination of statistical analysis, feature engineering, and Long Short-Term Memory (LSTM) neural networks, the project identifies complex temporal patterns and correlations between pollutants and meteorological variables. This work establishes a foundation for proactive air quality management and public health protection.

## Key Features

**Data Preparation:** Rigorous preprocessing, including anomaly removal, imputation of missing values using K-Nearest Neighbors (KNN), and aggregation to daily resolution.
**Feature Engineering:** Incorporation of additional variables such as holidays, weekends, and recurring temporal cycles derived from periodograms to enhance predictive power.
**Exploratory Data Analysis: **Identification of trends, seasonality, and key correlations through visualization and statistical tests (KPSS and ADF) to validate data stationarity.
**Model Implementation:** Deployment of an LSTM neural network, optimized for capturing short- and long-term dependencies in time-series data, with tailored architecture and hyperparameter settings.

## Why This Matters

Air pollution in Monterrey is a pressing issue, primarily driven by industrial activity, vehicular emissions, and local meteorological conditions. This repository represents a step towards leveraging modern machine learning techniques to predict pollution levels, inform mitigation strategies, and protect vulnerable populations.
