# Project 3

  - Project Description: [Project3_2021.pdf](https://github.com/giannhskp/Software-Development-for-Algorithmic-Problems_Project-3/blob/master/Project3_2021.pdf)
  - Project Documentation: [README.pdf](https://github.com/giannhskp/Software-Development-for-Algorithmic-Problems_Project-3/blob/master/README.pdf)

Project Topics:
  - Time Series Forecasting
  - Time Series Anomaly Detection
  - Time Series Compression
  - Time Series Clustering after compression

## Part 1 - Forecasting
Implementation: [forecast.py](https://github.com/giannhskp/Software-Development-for-Algorithmic-Problems_Project-3/blob/master/forecast.py)

We implemented a recursive Neural Network with stacked LSTM layers that forecasts the expected values of time series.

The RNN receives as input a WINDOW of values and forecasts the next expected value.
The training was performed using the the given dataset: [nasdaq2007_17.csv](https://github.com/giannhskp/Software-Development-for-Algorithmic-Problems_Project-3/blob/master/input_files/nasdaq2007_17.csv).
The final model was trained using all the time series of the dataset. We also trained 20 different models using only one time series as training dataset for each model

The final model structure, the results and the evalutation of the trained model are presented in the [Project Documentation](https://github.com/giannhskp/Software-Development-for-Algorithmic-Problems_Project-3/blob/master/README.pdf).

## Part 2 - Anomaly Detection
Implementation: [detect.py](https://github.com/giannhskp/Software-Development-for-Algorithmic-Problems_Project-3/blob/master/detect.py)

We implemented an auto-encoder using a recursive Neural Network with stacked LSTM layers. The RNN contains both encoding layers and decoding layers.

By using the Mean Absolute Error (MAE) and after deciding the proper threshold, the RNN is capable of detecting all the anomalies in a given time series.
All the anomalies of a time series are presented using a graph plot.
The training was performed using the the given dataset: [nasdaq2007_17.csv](https://github.com/giannhskp/Software-Development-for-Algorithmic-Problems_Project-3/blob/master/input_files/nasdaq2007_17.csv).

The final model structure, the results and the evalutation of the trained model are presented in the [Project Documentation](https://github.com/giannhskp/Software-Development-for-Algorithmic-Problems_Project-3/blob/master/README.pdf).

## Part 3 - Time Series Compression
Implementation: [reduce.py](https://github.com/giannhskp/Software-Development-for-Algorithmic-Problems_Project-3/blob/master/reduce.py)

We implemented a convolutional autoencoder Neural Network, containing both encoding layers and decoding layers with the proper bottleneck.

The goal of the NN is to compress a time series. Every time series is splitted in sets of 10 not-overlapping consecutive values and given as input to the NN. 
The 10-valued sets are compressed into 3 valued sets by the NN. After combining the 3-valued sets, the final compressed time series is produced.

The script receives 2 input files containing time series and produces 2 corresponding files that contain the compressed time series.

The final model structure, the results and the evalutation of the trained model are presented in the [Project Documentation](https://github.com/giannhskp/Software-Development-for-Algorithmic-Problems_Project-3/blob/master/README.pdf).

## Part 4 - Clustering after compression
After creating the compressed time series we compare the clustering procedure between the original dataset and the compressed dataset.

We use the clustering algorithms that were implemented in [Project2](https://github.com/giannhskp/Software-Development-for-Algorithmic-Problems_Project-2). We also compare the datasets in the Nearest Neighbor search.

The results, the comparison between the two approaches and the final conclutions are presented in the [Project Documentation](https://github.com/giannhskp/Software-Development-for-Algorithmic-Problems_Project-3/blob/master/README.pdf).
