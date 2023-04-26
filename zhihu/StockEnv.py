# Reference:
# zhihu: https://zhuanlan.zhihu.com/p/460939061?utm_id=0
# FinRL: https://github.com/AI4Finance-Foundation/FinRL/blob/master/finrl/meta/env_stock_trading/env_stocktrading.py

import random

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gymnasium as gym
from gymnasium import spaces



class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 df,
                 stock_dim, 
                 hmax,
                 tech_indicator_list, 
                 buy_cost_pct=1e-3,
                 sell_cost_pct=1e-3,
                 reward_scaling=2**-11,
                 initial_amount=None,
                 turbulence_threshold=None,
                 risk_indicator_col="turbulence",
                 initial=True,
                 print_verbosity=10):
        
        super(StockTradingEnv, self).__init__()

        self.print_verbosity = print_verbosity




        self.df = df
        self.date_array = self.df.date.unique().tolist()
        self.stock_dim = stock_dim
        self.tech_indicator_list = tech_indicator_list
        self.hmax = hmax
        self.risk_indicator_col = risk_indicator_col
        self.turbulence_threshold = turbulence_threshold

        # data at current day
        self.day = 0
        self.data = self.df.loc[self.df["date"] == self.date_array[self.day], :]
        self.num_stock_shares = [0] * self.stock_dim

        self.initial_amount = initial_amount

        # environment information
        self.action_dim = self.stock_dim
        # state dimension: balance + (adj.closeprice,share) * stock_dim + tech_index_dim * stock_dim
        self.state_dim = 1 + 2 * self.stock_dim + len(tech_indicator_list) * self.stock_dim

        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling

        # action: (-k, k)
        # action shape: num of stocks
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.action_dim,), dtype=np.float32
        )

        # state space dim: 
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)

        # reset
        self.initial = initial
        self.state = self._initiate_state()
        
        # initiate
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0

        # reward everyday
        self.rewards_memory = []
        # action everyday
        self.actions_memory = []
        # state
        self.state_memory = []
        # date 
        self.date_memory = [self.date_array[self.day]] # initiate date to first day


        


        

        
    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
    
    def _sell_stock(self, index, action):
        """
        index: which stock
        """
        def _do_sell_normal():
            # if (
            #     self.state[1 + 2 * self.stock_dim + index ] != True # check first technical index
            # ):  # check if the stock is able to sell, for simlicity we just add it in techical index
            if self.state[index + 1] > 0: 
                # Sell only if the price is > 0 (no missing data in this particular date)
                # perform sell action based on the sign of the action
                if self.state[index + self.stock_dim + 1] > 0:
                    # print(f"Num shares before: {self.state[index+self.stock_dim+1]}")

                    # Sell only if current asset is > 0
                    sell_num_shares = min(
                        abs(action), self.state[index + self.stock_dim + 1]
                    )
                    sell_amount = (
                        self.state[index + 1]
                        * sell_num_shares
                        * (1 - self.sell_cost_pct)
                    )
                    # update balance
                    self.state[0] += sell_amount

                    self.state[index + self.stock_dim + 1] -= sell_num_shares
                    self.cost += (
                        self.state[index + 1]
                        * sell_num_shares
                        * self.sell_cost_pct
                    )
                    self.trades += 1
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = 0

            return sell_num_shares

        # perform sell action based on the sign of the action
        if self.turbulence_threshold is not None:
            if self.turbulence >= self.turbulence_threshold:
                if self.state[index + 1] > 0:
                    # Sell only if the price is > 0 (no missing data in this particular date)
                    # if turbulence goes over threshold, just clear out all positions
                    if self.state[index + self.stock_dim + 1] > 0: # hold share > 0
                        # Sell only if current asset is > 0
                        sell_num_shares = self.state[index + self.stock_dim + 1]
                        sell_amount = (
                            self.state[index + 1]
                            * sell_num_shares
                            * (1 - self.sell_cost_pct)
                        )
                        # update balance
                        self.state[0] += sell_amount
                        self.state[index + self.stock_dim + 1] = 0
                        self.cost += (
                            self.state[index + 1]
                            * sell_num_shares
                            * self.sell_cost_pct
                        )
                        self.trades += 1
                    else:
                        sell_num_shares = 0
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = _do_sell_normal()
        else:
            sell_num_shares = _do_sell_normal()

        return sell_num_shares

    def _buy_stock(self, index, action):
        def _do_buy():
            # if (
            #     self.state[index + 2 * self.stock_dim + 1] != True
            # ):  # check if the stock is able to buy
            if self.state[index + 1] >0:
                # Buy only if the price is > 0 (no missing data in this particular date)
                available_amount = self.state[0] // (
                    self.state[index + 1] * (1 + self.buy_cost_pct)
                )  # when buying stocks, we should consider the cost of trading when calculating available_amount, or we may be have cash<0
                # print('available_amount:{}'.format(available_amount))

                # update balance
                buy_num_shares = min(available_amount, action)
                buy_amount = (
                    self.state[index + 1]
                    * buy_num_shares
                    * (1 + self.buy_cost_pct)
                )
                self.state[0] -= buy_amount

                self.state[index + self.stock_dim + 1] += buy_num_shares

                self.cost += (
                    self.state[index + 1] * buy_num_shares * self.buy_cost_pct
                )
                self.trades += 1
            else:
                buy_num_shares = 0

            return buy_num_shares

        # perform buy action based on the sign of the action
        if self.turbulence_threshold is None:
            buy_num_shares = _do_buy()
        else:
            if self.turbulence < self.turbulence_threshold:
                buy_num_shares = _do_buy()
            else:
                buy_num_shares = 0
                pass

        return buy_num_shares

    def step(self, actions):
        # terminate when reach lastest day
        self.terminal = self.day >= len(self.df.date.unique()) - 1

        if self.terminal:
            # print(f"Episode: {self.episode}")
            
            end_total_asset = self.state[0] + sum( #cash amount, not include capital, initial_amount is only cash part of our initial asset
                np.array(self.state[1 : (self.stock_dim + 1)]) # price per stock
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]) # hold share per stock
            )
            df_total_value = pd.DataFrame(self.asset_memory) 
            tot_reward = (
                end_total_asset
                - self.asset_memory[0] # first day value
            )  
            df_total_value.columns = ["account_value"]
            df_total_value["date"] = self.date_memory
            df_total_value["daily_return"] = df_total_value["account_value"].pct_change(1) # pct change with time difference = 1
            # if df_total_value["daily_return"].std() != 0:
            #     sharpe = (
            #         (252**0.5)
            #         * df_total_value["daily_return"].mean()
            #         / df_total_value["daily_return"].std()
            #     )
            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.columns = ["account_rewards"]
            df_rewards["date"] = self.date_memory[:-1]
            if self.episode % self.print_verbosity == 0:
                print(f"day: {self.day + 1}, episode: {self.episode}")
                print(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
                print(f"end_total_asset: {end_total_asset:0.2f}")
                print(f"total_reward: {tot_reward:0.2f}")
                print(f"total_cost: {self.cost:0.2f}")
                print(f"total_trades: {self.trades}")
                # if df_total_value["daily_return"].std() != 0:
                #     print(f"Sharpe: {sharpe:0.3f}")
                print("=================================")

                df_actions = self.save_action_memory()
                df_actions.to_csv(
                        "results/actions_{}.csv".format(
                            self.episode               
                            )
                    )
                df_total_value.to_csv(
                        "results/account_value_{}.csv".format(
                            self.episode
                        ),
                        index=False,
                    )
                df_rewards.to_csv(
                        "results/account_rewards_{}.csv".format(
                            self.episode
                        ),
                        index=False,
                    )
                plt.plot(self.asset_memory, "r")
                plt.savefig(
                        "results/account_value_{}.png".format(
                    self.episode
                                            )
                    )
                plt.close()


            
            return self.state, self.reward, self.terminal, {}

        else:
            actions = actions * self.hmax  # actions initially is scaled between 0 to 1 , action of each stock
            actions = actions.astype(
                int
            )  # convert into integer because we can't by fraction of shares
            if self.turbulence_threshold is not None:
                if self.turbulence >= self.turbulence_threshold:
                    actions = np.array([-self.hmax] * self.stock_dim)
            begin_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            # print("begin_total_asset:{}".format(begin_total_asset))

            argsort_actions = np.argsort(actions) # index of ascending value
            sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]] # index of sell (-k,-1)
            buy_index = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]] # buy (1,k)

            for index in sell_index:
                # print(f"Num shares before: {self.state[index+self.stock_dim+1]}")
                # print(f'take sell action before : {actions[index]}')
                actions[index] = self._sell_stock(index, actions[index]) * (-1) # number of shares sold for each stock
                # print(f'take sell action after : {actions[index]}')
                # print(f"Num shares after: {self.state[index+self.stock_dim+1]}")

            for index in buy_index:
                # print('take buy action: {}'.format(actions[index]))
                actions[index] = self._buy_stock(index, actions[index])

            self.actions_memory.append(actions)

            # state: s -> s+1
            self.day += 1
            self.data = self.df.loc[self.df["date"] == self.date_array[self.day], :]
            if self.turbulence_threshold is not None:
                if len(self.df.tic.unique()) == 1:
                    self.turbulence = self.data[self.risk_indicator_col]
                elif len(self.df.tic.unique()) > 1:
                    self.turbulence = self.data[self.risk_indicator_col].values[0]
            self.state = self._update_state()

            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            self.asset_memory.append(end_total_asset)
            self.date_memory.append(self.date_array[self.day])
            self.reward = end_total_asset - begin_total_asset
            self.rewards_memory.append(self.reward)
            self.reward = self.reward * self.reward_scaling
            self.state_memory.append(
                self.state
            )  # add current state in state_recorder for each day

        return self.state, self.reward, self.terminal, {}

    def reset(self):
        # initiate state
        self.state = self._initiate_state()

        self.asset_memory = [
            self.initial_amount + np.sum(
                    np.array(self.num_stock_shares)
                    * np.array(self.state[1 : 1 + self.stock_dim])
                )
            ]

        self.day = 0
        self.data = self.df.loc[self.df["date"] == self.date_array[self.day], :]
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False
        # self.iteration=self.iteration
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self.date_array[self.day]] # initiate date to first day

        self.episode += 1

        return self.state

    
    # New
    def _initiate_state(self):
        if self.initial:
            # For Initial State
            # if len(self.df.tic.unique()) > 1:

            # for multiple stock
            tech_indicator_every_stock = []
            for tech in self.tech_indicator_list:
                tech_indicator_every_stock += self.data[tech].values.tolist() # [tech index stock1, tech index stock2, tech index stock3]

            state = (
                [self.initial_amount]
                + self.data.close.values.tolist() # [close1, close2]
                + self.num_stock_shares # [share1, share2]
                + tech_indicator_every_stock
            )  
            
            # else:
            #     # for single stock
            #     tech_indicator_every_stock = []
            #     for tech in self.tech_indicator_list:
            #         tech_indicator_every_stock += self.data[tech].values.tolist() # [tech index stock1, tech index stock2, tech index stock3]

            #     state = (
            #         [self.initial_amount]
            #         + self.data.close.values.tolist()
            #         + [0] * self.stock_dim
            #         + tech_indicator_every_stock
            #     )
        return state
    
    # New
    def _update_state(self):
        # if len(self.df.tic.unique()) > 1:

        # for multiple stock
        tech_indicator_every_stock = []
        for tech in self.tech_indicator_list:
            tech_indicator_every_stock += self.data[tech].values.tolist()
        state = (
            [self.state[0]]
            + self.data.close.values.tolist() # dim same as stock_dim, price
            + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]) # length = stock_dim, share
            + tech_indicator_every_stock
        )

        # else:
        #     # for single stock
        #     state = (
        #         [self.state[0]]
        #         + self.data.close.values.tolist()
        #         + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
        #         + [self.data[tech].values for tech in self.tech_indicator_list]
        #     )

        return state
    
    # add save_state_memory to preserve state in the trading process
    def save_state_memory(self):
        if len(self.df.tic.unique()) > 1:
            # date and close price length must match actions length
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]

            state_list = self.state_memory
            df_states = pd.DataFrame(
                state_list,
                columns=[
                    "cash",
                    "Bitcoin_price",
                    "Gold_price",
                    "Bitcoin_num",
                    "Gold_num",
                    "Bitcoin_Disable",
                    "Gold_Disable",
                ],
            )
            df_states.index = df_date.date
            # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        else:
            date_list = self.date_memory[:-1]
            state_list = self.state_memory
            df_states = pd.DataFrame({"date": date_list, "states": state_list})
        # print(df_states)
        return df_states

    def save_asset_memory(self):
        date_list = self.date_memory
        asset_list = self.asset_memory
        # print(len(date_list))
        # print(len(asset_list))
        df_account_value = pd.DataFrame(
            {"date": date_list, "account_value": asset_list}
        )
        return df_account_value

    def save_action_memory(self):
        if len(self.df.tic.unique()) > 1:
            # date and close price length must match actions length
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]

            action_list = self.actions_memory
            df_actions = pd.DataFrame(action_list)
            df_actions.columns = self.data.tic.values
            df_actions.index = df_date.date
            # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        else:
            date_list = self.date_memory[:-1]
            action_list = self.actions_memory
            df_actions = pd.DataFrame({"date": date_list, "actions": action_list})
        return df_actions


    
   
