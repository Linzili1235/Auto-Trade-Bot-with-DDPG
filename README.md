**OUTLINE OF THIS REPO:**

```
train_notebook.ipynb / test_notebook.ipynb:

    Part 1. Data Crawling -> yahoodownloader.py
    
                          -> config_tickers.py
    
                          -> preprocessors.py                      

    Part 2. Stock Environment -> StockEnv.py

    Part 3. DDPG Model -> model.py
                        
                       -> ReplayBuffer.py
```
<h2 align="center"> Deep Deterministic Policy Gradient for Automated Stock Trading </h2>

<h3> 1. Introduction </h3>

Automated stock trading has been a popular area of research for decades due to its potential to generate profits in the financial markets, and with the rise of deep learning and reinforcement learning techniques, new possibilities have emerged for creating more effective trading strategies. In recent years, deep reinforcement learning has shown promising results in automated stock trading, leading to improved performance and better returns. However, there are still challenges to overcome in creating robust and reliable models that can adapt to changing market conditions, as it involves many factors that affect stock price changes, such as economic indicators, company financial reports, political events, etc. In this paper, we explore a strategy for automated stock trading using deep reinforcement learning techniques. And the success of the automated trading strategy depends on the quality of the selected model and data preprocessing, otherwise it can be problematic. Therefore, automated trading through deep reinforcement learning is a challenging and promising research direction. By combining multiple strategies, we aim to create a more robust and reliable system that adapts to different market conditions and generates stable returns. In this paper, we present our methodology and results and discuss the implications of our findings for future research in this area.

Other research groups have conducted similar studies in the past, but their work has typically been based on traditional statistical methods, simple datasets or conventional machine learning algorithms, which have some difficulties in dealing with high-dimensional, nonlinear financial data. In contrast, our research problem implements automated trader robot using deep reinforcement learning algorithms with Deep Deterministic Policy Gradient (DDPG), designs deep reinforcement learning models to predict stock price changes, and how to integrate multiple strategies into an overall trading strategy to improve the accuracy and profitability of the trading strategy, and uses integrated learning strategies to further improve performance.



<h3> 2. Method </h3>

We model stock trading as a Markov Decision Process (MDP), and formulate our trading objective as a maximization of expected return.

**2.1 Stock Market Environment**

We build the environment to simulate the real-world stock market before training the deep reinforcement learning agent. In practical trading, investors need various information, such as current prices, historical prices, current holding shares, to make investment decisions. In our experiment, similar information is integrated into the simulated stock market for our agent to take actions.

We use a continuous action space to model the trading of multiple stocks. We assume that our portfolio has 2 stocks in total, max number of shares traded per day is 100 and the initial balance is 0.2 million dollars. 

**1) State Space:** We use a 53-dimensional vector consists of twenty-seven parts of information to represent the state space of multiple stocks trading environment: [bt, pt, pt-1, pt-2, pt-3, pt-4, ht]. (All output state will be normalized before putting into the reinforcement learning algorithm.) Each component is defined as follows: 

    - bt ∈ R+: available account balance at current time step t.
    - pt ∈ R5+ : open price, high price, low price, adjusted close price, volume of each stock at current time step t. Same logic for pt-1, pt-2, pt-3, pt-4.
    - ht ∈ Z2+ : shares owned of each stock at time step t. 

**2) Action Space:** For one single stock, the action space is defined as {−k, ..., −1, 0, 1, ..., k}, where k and −k presents the number of shares we can buy and sell, and k ≤ hmax while hmax is a predefined parameter that sets as the maximum amount of shares for each buying and selling action. Therefore the size of the entire action space is (2k+1)*2 . The action space is then normalized to [−1, 1], since the RL algorithms will output normalized action.

**3) Trading Constraint:** We integrate transaction cost when calculating returns from investment actions, which is consistent with the real-life stock trading setting. In practical trading, there are various transaction costs such as exchange fees, execution fees and so on. In our experient, we set our transaction cost to be 0.1% of the value of each transaction (either buy or sell). ct = pTkt × 0.1%.

**4) State transition:** At each state, one of three possible actions is taken on stock d in the portfolio.

    - Selling: if selling k[d] shares, then ht+1[d] = ht[d] − k[d], where k[d] ∈ [1, h[d]]  and d = 1, ..., D.

    - Holding: ht+1[d] = ht[d].

    - Buying: if buying k[d] shares, then ht+1[d] = ht[d] + k[d], , where k[d] ∈ [1, hmax]  and d = 1, ..., D.

Balance: After executing buying and selling actions, bt+1 = bt + (pdt ht+1,sell[d] - ctd) - (pdt ht+1,buyl[d] + ctd), and d = 1, ..., D.

**5) Total Asset:** total asset owned everyday zt = bt + pdtht[d], where d = 1, ..., D.

**6) Reward:** rt = zt - zt-1. After normalization, the final reward is adjusted by reward mechanism, which relates to the return rt / zt-1. The higher return, bigger value will be added to the normalized reward to encourage positive return. The lower return, bigger value will be deducted to the normalized reward to punish negative return.  


**2.2 Model**

Deep reinforcement learning is a powerful technique to solve multi-period decision making problems arising in quantitative finance. In this work, we use one of the most popular RL algorithms for solving high-dimensional RL problems with continuous states and actions, the deep deterministic policy gradient (DDPG) algorithm. 

The mechanisms of DDPG can be simply divided into two parts: Interacting with environment and optimizing the network. For the first part, the agent interacts with the stock environment which we introduced above to collect data samples including the current state, action with Gaussian noises, next state, reward, and whether done, and stores them in the replay buffer. Once the sample size within the replay buffer is large enough to perform a training process as a batch, the second part will be automatically triggered. 

<p align="center">
<img src=plots/model_sampling.png width="800"/>
<br> Figure 1: Interact with the environment.
</p>


For the second part, there are four networks involved: Actor, Critic, Actor Target, and Critic Target. The first pair of actor and critic networks are used to predict the Q-value for the current state. The other pair of target networks are used to compute the target Q-value for the next state. With these two Q-values as well as the current reward, TD-error can be leveraged to update the critic network, while the parameters of actor network will be optimized by policy gradient. Then, soft update is applied on two target networks, a similar process to the TD-error method. As these four networks keep optimized during the long training process, the model will eventually tend to converge, which results in a series of steady total rewards.

<p align="center">
<img src=plots/model_optimizing.png width="800"/>
<br> Figure 2: Optimize the networks.
</p>



We select two stocks Goldman Sachs (GS) and UnitedHealth Group (UNH) as our representatives and retrieve daily stock data from Yahoo Finance API, year 2018 to year 2019 for train and year 2020 for test. 



<h3> 3. Results and Discussion </h3>

To evaluate the performance of DDPG algorithm on this stock trading problem, we make comparisons with the Buy-and-Hold policy, which is always seen as a traditional optimal policy and find that the total reward of DDPG is significantly larger than that of Buy-and-Hold, as shown in Table 1. Thus, DDPG is proved to make very powerful prediction on the stock trading. 

<p align="center">Table 1: Total rewards ($) earned by DDPG and Buy-and-Hold Policy.</p>

<table align="center">
    <thead>
        <tr>
            <th align="left"></th>
            <th align="center">Total Reward - DDPG</th>
            <th align="center">Total Reward - Buy-and-Hold</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td align="center">Train</td>
            <td align="center">0.6M</td>
            <td align="center">0.08M</td>
        </tr>
        <tr>
            <td align="center">Test</td>
            <td align="center">0.8M</td>
            <td align="center">0.04M</td>
        </tr>
    </tbody>
</table>



The reason why the agent makes much more money in testing than training using the same optimal model is probably that the stock price has been supremely volatile during the testing period, which provides the agent more opportunity to achieve higher total rewards. Additionally, the changes of account asset within one episode for both train and test are plotted in Figure 3, where the testing curve shows a greater variance than training.

<p align="center">
<img src=plots/train_asset.png width="400"/> 
<img src=plots/test_asset.png width="400"/>
<br> Figure 3: Account asset change within one episode.
</p>



Since the core of DDPG algorithm is neural network, it is extremely hard to interpret our optimal policy. Thus, the changes of number of shares held by the agent within one episode for both train and test are displayed in Figure 4 to visualize the actions that the agent takes when it comes to different situations. 

<p align="center">
<img src=plots/train_share.png width="400"/> 
<img src=plots/test_share.png width="400"/>
<br> Figure 4: Number of shares held by the agent within one episode.
</p>

Furthermore, the daily closing prices are shown in Figure 5 as supplementary to help understand and evaluate whether the agent makes wise move. For both train and test processes, we can observe very similar patterns before and after the lowest point of price. Therefore, simply take the train result as an example to illustrate these interesting and meaningful patterns. The GS reaches the lowest point and the UNH coincedently meets a sharp decrease around the 230th trading day, during which period the agent shows so much caution that the number of shares held has remained relatively constant. Before the lowest point, the price of GS decreases in general while that of UNH shows an overally increasing tendency, which leads the agent to put most of the investments on the UNH. On the contrary, our agent keeps investing vastly after the lowest point and achieves a considerable profit in the end.

<p align="center">
<img src=plots/train_price.png width="400"/> 
<img src=plots/test_price.png width="400"/>
<br> Figure 5: Closing Prices Change by Tickers by Trading Days.
</p>




<h3> 4. Conclusion </h3>

In conclusion, our study on automated stock trader robots using deep reinforcement learning with DDPG has provided valuable insights into the potential of machine learning techniques in financial markets. By analyzing historical stock data and developing a deep reinforcement learning model, we were able to achieve impressive results in terms of accuracy and profitability. Our study has shown that deep reinforcement learning can be a powerful tool for automated stock trading, but it also highlights the need for further research in this area.

The findings of our project have important implications for investors, financial analysts, and policymakers. By leveraging the power of machine learning, we can make more informed investment decisions and potentially increase returns. However, it is also important to recognize the potential risks and limitations of these techniques.

Overall, this study demonstrates the potential of machine learning in financial markets and provides a foundation for further research and development. By continuing to explore the capabilities and limitations of these techniques, we can improve our understanding of financial markets and make more informed decisions in the future.





