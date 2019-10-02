# LSTM_Anomaly_Detection

This repo contains the analysis of a gaming firm's production logs.
Using this data I build models to predict if a user will install a product after being shown an advertisement.

The analysis starts with traditional models used for predicting clicks. Then switch over to treating the problem as an anomaly detection using a LSTM.
To learn the latent space of observation that don't clicks

I initially start with testing and comparing individual models then move on to ensemble models and finally a LSTM.

Data is withheld for privacy of firm but you are free to run the model with your own data that match the features of the model.
To run model give your columns the same names as the ones used in the analysis and execute the following command:

```python test_data.csv```

The results will be a csv with the following columns

```id: site visitor id```

```install_prob: probability the user decides to install```

Confusion matrix of random forest models

![]()

Model parameters:

``````

Chart of LSTM results

Observations 1 to X  is the model learning the latent space of observations that do not choose to install after being shown an ads. (Training)
Observations X to N are mix of observations that do and do not install after being shown adds (Testing).
The spikes in the chart from observations X to N are are the model predicts for visitors who will choose to install after being shown an add.
![]()
![]()
