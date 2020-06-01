# ADL HW2
## How to train model
* Go to src/
* Specific training data path and validation data path in first few line of code
* python3.7 buildDataset.py, it will generate dataset at ../data 
* python3.7 train.py 
## How to plot figure for culmulative answer length
* Go to src/
* Specific training data path in first few line of code
* python3.7 plot_ans_len.py 
## How to plot figure for answerable threshold
* Go to src/
* Result for my experiment have already store in directory result_for_fig
* python3.7 plot_answerable_threshold.py 
## How to predict
* python3.7 predict.py --test_data_path testDataPath --output_path outputPath 
