import numpy as np
import pandas as pd
import re
import random
from datetime import datetime
import json
from argparse import Namespace
import os

STARTING_NET_WORTH = 1_000_000
STARTING_CASH_RATIO = 0.5
# STARTING_CASH_RATIO = 1
GAS_LIMITS = 21000  # unit
GAS_PRICE = 70  # gwei
GAS_FEE = GAS_LIMITS * GAS_PRICE * 1e-9  # eth per txn
EX_RATE = 4e-3  # exchange fee = txn_amount * ex_rate
# GAS_FEE = 0
# EX_RATE = 0
SMA_PERIODS = [5, 10, 15, 20, 30]


def get_paths(args):
    dataset = args.dataset.lower()
    price_dict = {'eth': 'eth_daily.csv', 'btc': 'bitcoin_daily_price.csv', 'sol': 'solana_daily_price.csv'}
    txn_dict = {'eth': 'eth_more_transaction_statistics.csv', 'btc': 'bitcoin_transaction_statistics.csv', 'sol': 'solana_transaction_statistics.csv'}
    news_dict = {'eth': 'gnews', 'btc': 'selected_bitcoin_202301_202401', 'sol': 'selected_solana_202301_202401'}
    timecol_dict = {'eth': 'snapped_at', 'btc': 'timeOpen', 'sol': 'timeOpen'}
    price_timefmt_dict = {'eth': "%Y-%m-%d %H:%M:%S UTC", 'btc': "%Y-%m-%dT%H:%M:%S.%fZ", 'sol': "%Y-%m-%dT%H:%M:%S.%fZ"}
    txn_timefmt_dict = {'eth': "%d/%m/%y %H:%M", 'btc': "%Y-%m-%d %H:%M:%S.%f UTC", 'sol': "%Y-%m-%d %H:%M:%S.%f UTC"}
    return f'data/{price_dict[dataset]}', f'data/{txn_dict[dataset]}', f'data/{news_dict[dataset]}', timecol_dict[dataset], price_timefmt_dict[dataset], txn_timefmt_dict[dataset]


class ETHTradingEnv:
    def __init__(self, args):
        price_path, txn_path, self.news_dir, self.timecol, self.price_timefmt, txn_timefmt = get_paths(args)
        starting_date, ending_date = args.starting_date, args.ending_date
        df = pd.read_csv(price_path)
        df = df.sort_values(self.timecol)
        df['date'] = pd.to_datetime(df[self.timecol], format=self.price_timefmt)
        
        # SMA
        for period in SMA_PERIODS:
            df[f'SMA_{period}'] = df['open'].rolling(window=period).mean()
            df[f'STD_{period}'] = df['open'].rolling(window=period).std()
        # MACD and Signal Line
        df['EMA_12'] = df['open'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['open'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        self.data = df[(df['date'] >= starting_date) & (df['date'] <= ending_date)]  # only use ending_date for open price
        
        self.txn_stat = pd.read_csv(txn_path).sort_values('day')
        self.txn_stat['date'] = pd.to_datetime(self.txn_stat['day'], format=txn_timefmt)
        self.total_steps = len(self.data)
        self.starting_net_worth = STARTING_NET_WORTH
        self.starting_cash_ratio = STARTING_CASH_RATIO
        # self.reset()

    def get_close_state(self, today, next_day):
        next_open_price = next_day['open']
        close_net_worth = self.cash + self.eth_held * next_open_price
        close_roi = close_net_worth / self.starting_net_worth - 1  # return on investment
        today_roi = close_net_worth / self.last_net_worth - 1
        self.last_net_worth = close_net_worth

        date = today[self.timecol]
        parsed_time = datetime.strptime(date, self.price_timefmt)
        year, month, day = parsed_time.year, parsed_time.month, parsed_time.day

        # Fixed: Use today's technical indicators (decision made at today's close)
        ma5 = today['SMA_5']
        ma10 = today['SMA_10']
        ma15 = today['SMA_15']
        ma20 = today['SMA_20']
        slma_signal = 'hold'
        short_ma = ma15
        long_ma = ma20
        if short_ma > long_ma:
            slma_signal = 'sell'
        elif short_ma < long_ma:
            slma_signal = 'buy'

        sma = today[f'SMA_20']
        sd = today[f'STD_20']
        multiplier = 2
        upper_band = sma + (sd * multiplier)
        lower_band = sma - (sd * multiplier)
        today_open_price = today['open']
        boll_signal = 'hold'
        if today_open_price < lower_band:
            boll_signal = 'buy'
        elif today_open_price > upper_band:
            boll_signal = 'sell'

        macd = today['MACD']
        macd_signal_line = today['Signal_Line']
        macd_signal = 'hold'
        if macd < macd_signal_line:
            macd_signal = 'buy'
        elif macd > macd_signal_line:
            macd_signal = 'sell'

        # today's txn stats
        txn_stat = self.txn_stat[self.txn_stat['date'] == parsed_time]
        txn_cols = set(self.txn_stat.columns.tolist()) - set(['date', 'day'])
        if txn_stat.empty:
            txn_data = {col: 'N/A' for col in txn_cols}
        else:
            txn_data = {col: txn_stat[col].values[0] for col in txn_cols}

        # today's news
        news_path = f"{self.news_dir}/{year}-{str(month).zfill(2)}-{str(day).zfill(2)}.json"
        if not os.path.exists(news_path):
            news = 'N/A'
        else:
            loaded_news = json.load(open(news_path))
            seen_titles = set()  # remove duplicate
            news = []
            for loaded_item in loaded_news:
                if loaded_item['title'] not in seen_titles:
                    item = {k: loaded_item[k] for k in ['id', 'time', 'title', 'content']}  # omit url
                    K = 5000  # clip characters
                    if len(item['content']) > K:
                        item['content'] = item['content'][:K] + '...' 
                    news.append(item)
                    seen_titles.add(item['title'])

        close_state = {  # selectively used in prompt
            'cash': self.cash,
            'eth_held': self.eth_held,
            'open': today['open'],
            'net_worth': close_net_worth,
            'roi': close_roi,
            'today_roi': today_roi,
            'technical': {
                # 'short_long_ma_signal': slma_signal,
                'macd_signal': macd_signal,
                # 'bollinger_bands_signal': boll_signal,
            },
            'txnstat': txn_data,
            'news': news,
            'date': date,
        }
        return close_state

    def reset(self):
        self.current_step = 0
        today = self.data.iloc[self.current_step]
        if self.total_steps > 1:
            next_day = self.data.iloc[self.current_step + 1]
        else:
            next_day = today
        self.starting_price = today['open']
        self.cash = self.starting_net_worth * STARTING_CASH_RATIO
        self.eth_held = (self.starting_net_worth - self.cash) / self.starting_price
        self.last_net_worth = self.starting_net_worth
        close_state = self.get_close_state(today, next_day)
        info = {
            'starting_cash': self.cash,
        }
        reward = 0
        self.done = False
        self.last_state = close_state
        return close_state, reward, self.done, info

    # the agent receives last state and reward, takes an action, and receives new state and reward.
    # last state: today's news, today's open price, cash, held ETH
    # last reward: yesterday's profit
    # action: buy, sell, or hold (decide at today's close)
    # new state: next day's news, next day's open price, cash, held ETH
    # new reward: next day's profit
    def step(self, action):
        raw_action = action
        if type(action) == str:
            # actions = re.findall(r"[-+]?\d*\.\d+|\d+", action)
            actions = re.findall(r"-?(?:0(?:\.\d{1})|1\.0)", action)

            if len(actions) == 0:
                print(f'ERROR: Invalid llm response: {action}. Set to no action.')
                action = 0.00
            elif len(actions) == 1:
                action = float(actions[0])
            else:
                # print(f'Multiple actions in llm response: {action}. Pick one action.')
                # action = float(actions[0])
                action = float(actions[-1])

        if not -1 <= action <= 1:
            print(f"ERROR: Invalid action: {action}. Set to no action.")
            action = 0.00

        today = self.data.iloc[self.current_step]
        if self.current_step + 1 >= self.total_steps:
            self.done = True
            info = {
                'raw_action': raw_action,
                'actual_action': action,
                'starting_cash': self.starting_net_worth,
                'ref_all_in': self.starting_net_worth / self.starting_price * today['open'],
                'today': today[self.timecol],
            }
            return self.last_state, 0, self.done, info

        next_day = self.data.iloc[self.current_step + 1]
        # Execute trade at next day's open (decision made at today's close)
        trade_price = next_day['open']

        if self.current_step + 2 >= self.total_steps:
            self.done = True
            info = {
                'raw_action': raw_action,
                'actual_action': action,
                'starting_cash': self.starting_net_worth,
                'ref_all_in': self.starting_net_worth / self.starting_price * trade_price,
                'today': today[self.timecol],
            }
            return self.last_state, 0, self.done, info

        next_next_day = self.data.iloc[self.current_step + 2]
        valuation_price = next_next_day['open']  # Use next_next_day's open for performance calculation

        # Sell: action in [-1, 0)
        if -1 <= action < 0 and self.eth_held > 0:
            eth_diff = abs(action) * self.eth_held
            cash_diff = eth_diff * trade_price
            # Calculate fees
            gas_fee_cost = GAS_FEE * trade_price
            exchange_fee = cash_diff * EX_RATE
            total_fee = gas_fee_cost + exchange_fee

            # Execute trade
            self.eth_held -= eth_diff
            self.cash += cash_diff - total_fee  # Receive cash minus fees

        # Buy: action in (0, 1]
        if 0 < action <= 1 and self.cash > 0:
            # Calculate desired purchase amount
            desired_cash = abs(action) * self.cash

            # Calculate fees for this transaction
            gas_fee_cost = GAS_FEE * trade_price
            exchange_fee = desired_cash * EX_RATE
            total_fee = gas_fee_cost + exchange_fee
            total_cost = desired_cash + total_fee

            # Only execute if we have enough cash (including fees)
            if self.cash >= total_cost:
                eth_diff = desired_cash / trade_price
                self.cash -= total_cost
                self.eth_held += eth_diff
            # else: insufficient cash, skip transaction

        self.current_step += 1
        if self.current_step >= self.total_steps - 2:
            self.done = True

        close_state = self.get_close_state(next_day, next_next_day)
        reward = close_state['roi'] - self.last_state['roi']  # reward = today's roi gain.
        self.last_state = close_state
        info = {
            'raw_action': raw_action,
            'actual_action': action,
            'starting_cash': self.starting_net_worth,
            'ref_all_in': self.starting_net_worth / self.starting_price * valuation_price,
            'today': today[self.timecol],
        }
        return close_state, reward, self.done, info
    
    def close(self):
        pass
