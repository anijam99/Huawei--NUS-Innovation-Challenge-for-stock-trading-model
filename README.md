# Huawei--NUS-Innovation-Challenge-for-stock-trading-model
Stock Trading Model created as part of Huawei NUS Competition

A simple prediction strategy. 
Execute the order every 5 minutes according to the prediction result. 
If the result y>=0, that means the price will probably rise. 
Then buy in at the start of this round and meanwhile delay the sell order to the next round.
If the result y<0, that means the price will probably fall. 
Then sell out at the start of this round and meanwhile delay the buy order to the next round.

Work flow:
1. Use tick2trainData.py to read original tickdata files and produce a single trainData file for training.
    The trainData file includes 11 columns. 
    Column y is the price change rate of the next 5 minutes.
    Column x1 to x10 are 10 factors used to predict y.
    output: trainData.csv

2. Use trainModel.py to train a prediction model.
    output: model.pkl

3. Use run.sh to call myModel_demo.py and execute the strategy.

More information regarding the competition and dataset used can be found here: https://www.sg-innovationchallenge.org/problem-description
