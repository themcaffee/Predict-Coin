# Predict-Coin


Gathers data from GDAX and uses an LSTM to predict values


### Gather data
python gather_data.py


### Predict Data using model
python predict.py


#### TODO

- Use price prediction to create reinforcement learning algorithm
  to maximize profits given a starting amount.

- Start at sometime like 2 months ago
- Train on data before that time
- Iterate through history and at each point decide whether to buy, hodl, or sell
- It would be best if the amount to buy, hold, or sell was also predicted but for
  now just used fixed price
- Buy if market is predicted to go up
- Hold if relatively flat
- Sell if going down
- Have caps for amount of transactions
- Need to take into account fees for buying/selling
- Need to also eventually need to take into account the delay of purchasing/selling