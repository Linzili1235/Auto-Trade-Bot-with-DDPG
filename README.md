```
train_notebook.ipynb / test_notebook.ipynb:

    Part 1. Data Crawling -> yahoodownloader.py
    
                          -> config_tickers.py
    
                          -> preprocessors.py                      

    Part 2. Stock Environment -> StockEnv.py

    Part 3. DDPG Model -> model.py
                        
                       -> ReplayBuffer.py
```

**Train Data:** GSUNH_2y_train.csv (a subset from train_data.csv)

**Test Data:** GSUNH_2y_test.csv (a subset from test_data.csv)

**Note:** The only difference between train_notebook.ipynb and test_notebook.ipynb is that they use different data.
