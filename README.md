NFL Wide Receiver Touchdown Projection Model

This repository contains a machine learning framework for projecting NFL wide receiver (WR) touchdowns (TDs) using only historical performance data. 
The model is designed to avoid data leakage, train on past seasons, and backtest on out-of-sample years to evaluate predictive accuracy.


The goal of this project is to build an interpretable, data-driven approach for forecasting wide receiver touchdown totals. 
By using features such as red zone usage, efficiency metrics, and year-to-year performance trends, the model aims to provide actionable projections for upcoming NFL seasons.


Temporal train/test split to ensure realistic forecasting

Feature engineering for red zone metrics (targets, receptions, conversion rates) and rolling averages.

Cross-validated model evaluation using metrics such as MAE, RMSE, and R².

