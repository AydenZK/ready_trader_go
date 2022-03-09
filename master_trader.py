# Copyright 2021 Optiver Asia Pacific Pty. Ltd.
#
# This file is part of Ready Trader Go.
#
#     Ready Trader Go is free software: you can redistribute it and/or
#     modify it under the terms of the GNU Affero General Public License
#     as published by the Free Software Foundation, either version 3 of
#     the License, or (at your option) any later version.
#
#     Ready Trader Go is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU Affero General Public License for more details.
#
#     You should have received a copy of the GNU Affero General Public
#     License along with Ready Trader Go.  If not, see
#     <https://www.gnu.org/licenses/>.
from ast import Or
import asyncio
import itertools
import numpy as np
import scipy.stats as st

from typing import List

from ready_trader_go import BaseAutoTrader, Instrument, Lifespan, MAXIMUM_ASK, MINIMUM_BID, Side


LOT_SIZE = 10
POSITION_LIMIT = 100
ARBITRAGE_POS_LIMIT = 100
TICK_SIZE_IN_CENTS = 100

class OrderBook:
    def __init__(self, sequence_number: int, ask_prices: List[int], ask_volumes: List[int], bid_prices: List[int], bid_volumes: List[int]):
        self.sequence_number = sequence_number
        self.ask_prices = ask_prices
        self.ask_volumes = ask_volumes
        self.bid_prices = bid_prices
        self.bid_volumes = bid_volumes
        self.best_bid = bid_prices[0]
        self.best_ask = ask_prices[0]
        self.best_bid_vol = bid_volumes[0]
        self.best_ask_vol = ask_volumes[0]

class Historical:
    def __init__(self):
        self.historical_prices = {
            Instrument.FUTURE: [], # Future
            Instrument.ETF: []  # ETF
        }
        self.move=0
        self.total=0
        self.n=0


    def update(self, instrument, price):
        self.historical_prices[instrument].append(price)
        if instrument==Instrument.ETF and len(self.historical_prices[instrument])>=2:
            self.total+=(self.historical_prices[instrument][-1]-self.historical_prices[instrument][-2])**2
            self.n+=1
            self.move=(self.total/self.n)**(1/2)
    @property
    def min_time(self):
        return min(
            len(self.historical_prices[Instrument.FUTURE]), 
            len(self.historical_prices[Instrument.ETF]))

    def min_len_to_execute(self, result, length=10, safe_result=0):
        if self.min_time >= length:
            return result
        return safe_result

    @property
    def history_future(self):
        return self.historical_prices[Instrument.FUTURE][:self.min_time]

    @property
    def history_etf(self):
        return self.historical_prices[Instrument.ETF][:self.min_time]

    @property
    def std_etf(self):
        return self.min_len_to_execute(np.std(self.history_etf), safe_result=1)

    @property
    def std_future(self):
        return self.min_len_to_execute(np.std(self.history_future), safe_result=1)

    @property
    def corr(self):
        return self.min_len_to_execute(np.corrcoef(self.history_etf, self.history_future)[0][1])
    
    @property
    def cov(self):
        return self.min_len_to_execute(self.corr * self.std_etf * self.std_future)

    @property
    def beta(self):
        """beta of stock against future"""
        return self.min_len_to_execute(self.cov / (self.std_future**2))
    

class AutoTrader(BaseAutoTrader):
    """Example Auto-trader.

    When it starts this auto-trader places ten-lot bid and ask orders at the
    current best-bid and best-ask prices respectively. Thereafter, if it has
    a long position (it has bought more lots than it has sold) it reduces its
    bid and ask prices. Conversely, if it has a short position (it has sold
    more lots than it has bought) then it increases its bid and ask prices.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, team_name: str, secret: str):
        """Initialise a new instance of the AutoTrader class."""
        super().__init__(loop, team_name, secret)
        self.order_ids = itertools.count(1)
        self.bids = set()
        self.future_bids = set()
        self.asks = set()
        self.future_asks = set()
        self.ask_id = self.ask_price = self.bid_id = self.bid_price = self.position = self.futures_position = 0
        self.historical = Historical()
        self.arbitrage_ids = set()

    def get_total_position(self):
        return self.position + self.futures_position

    def on_error_message(self, client_order_id: int, error_message: bytes) -> None:
        """Called when the exchange detects an error.

        If the error pertains to a particular order, then the client_order_id
        will identify that order, otherwise the client_order_id will be zero.
        """
        self.logger.warning("error with order %d: %s", client_order_id, error_message.decode())
        if client_order_id != 0:
            self.on_order_status_message(client_order_id, 0, 0, 0)

    def on_hedge_filled_message(self, client_order_id: int, price: int, volume: int) -> None:
        """Called when one of your hedge orders is filled, partially or fully.

        The price is the average price at which the order was (partially) filled,
        which may be better than the order's limit price. The volume is
        the number of lots filled at that price.

        If the order was unsuccessful, both the price and volume will be zero.
        """
        self.logger.info("received hedge filled for order %d with average price %d and volume %d", client_order_id,
                         price, volume)
        if client_order_id in self.future_asks:
            self.futures_position -= volume
        elif client_order_id in self.future_bids:
            self.futures_position += volume

    def on_order_book_update_message(self, instrument: int, sequence_number: int, ask_prices: List[int],
                                     ask_volumes: List[int], bid_prices: List[int], bid_volumes: List[int]) -> None:
        """Called periodically to report the status of an order book.

        The sequence number can be used to detect missed or out-of-order
        messages. The five best available ask (i.e. sell) and bid (i.e. buy)
        prices are reported along with the volume available at each of those
        price levels.
        """
        self.logger.info("received order book for instrument %d with sequence number %d", instrument,
                         sequence_number)
        # adds to historical prices (mid price)
        self.historical.update(instrument, np.mean([bid_prices[0], ask_prices[0]]))
        if instrument == Instrument.FUTURE:
            self.futures = OrderBook(sequence_number, ask_prices,ask_volumes, bid_prices, bid_volumes)
            new_bid_price, new_ask_price = self.price(np.mean([bid_prices[0], ask_prices[0]]))
            self.logger.info("Prices: %d, %d",new_bid_price, new_ask_price)
            if self.bid_id != 0 and new_bid_price not in (self.bid_price, 0):
                self.send_cancel_order(self.bid_id)
                self.bid_id = 0
            if self.ask_id != 0 and new_ask_price not in (self.ask_price, 0):
                self.send_cancel_order(self.ask_id)
                self.ask_id = 0

            if self.bid_id == 0 and new_bid_price != 0 and self.position+LOT_SIZE < POSITION_LIMIT:
                self.bid_id = next(self.order_ids)
                self.bid_price = new_bid_price
                self.send_insert_order(self.bid_id, Side.BUY, new_bid_price, LOT_SIZE, Lifespan.GOOD_FOR_DAY)
                self.bids.add(self.bid_id)

            if self.ask_id == 0 and new_ask_price != 0 and self.position-LOT_SIZE > -POSITION_LIMIT:
                self.ask_id = next(self.order_ids)
                self.ask_price = new_ask_price
                self.send_insert_order(self.ask_id, Side.SELL, new_ask_price, LOT_SIZE, Lifespan.GOOD_FOR_DAY)
                self.asks.add(self.ask_id)
            self.logger.info("Bids: %d, Asks %d", self.bids, self.asks)            
    
        elif instrument == Instrument.ETF:
            self.etfs = OrderBook(sequence_number, ask_prices, ask_volumes, bid_prices, bid_volumes)
            if self.futures.best_bid > self.etfs.best_bid:
                if 0 < self.etfs.best_ask_vol <= self.futures.best_bid_vol and self.etfs.best_ask < self.futures.best_bid:
                    full_trade_vol = min(ARBITRAGE_POS_LIMIT-self.position, ARBITRAGE_POS_LIMIT+self.futures_position, self.etfs.best_ask_vol)
                    trade_vol = full_trade_vol // 2
                    if trade_vol > 0:
                        next_id = next(self.order_ids)
                        log = {
                            "POSITION": self.position,
                            "FUTURES_POSITION": self.futures_position,
                            "FULL_TRADE_VOL": full_trade_vol,
                            "MAX_VOLUME": self.etfs.best_bid_vol,
                            "ACTION": f"BUY {trade_vol} ETF @{self.etfs.best_ask}, ID: {next_id}"
                        }
                        self.logger.info(f"CUSTOM LOG: {log}")
                        self.send_insert_order(next_id, Side.BUY, self.etfs.best_ask, trade_vol, Lifespan.FAK) # buy etf
                        self.bids.discard(self.bid_id)
                        self.send_cancel_order(self.bid_id)
                        self.bid_id = 0
                        self.bids.add(next_id)
                        self.arbitrage_ids.add(next_id)

            if self.etfs.best_bid > self.futures.best_bid:
                if 0 < self.etfs.best_bid_vol <= self.futures.best_ask_vol and self.futures.best_ask < self.etfs.best_bid:
                    full_trade_vol = min(ARBITRAGE_POS_LIMIT+self.position, ARBITRAGE_POS_LIMIT-self.futures_position, self.etfs.best_bid_vol)
                    trade_vol = full_trade_vol // 2
                    if trade_vol > 0:
                        next_id = next(self.order_ids)
                        log = {
                            "POSITION": self.position,
                            "FUTURES_POSITION": self.futures_position,
                            "FULL_TRADE_VOL": full_trade_vol,
                            "MAX_VOLUME": self.etfs.best_bid_vol,
                            "ACTION": f"SELL {trade_vol} ETF @{self.etfs.best_bid}, ID: {next_id}"
                        }
                        self.logger.info(f"CUSTOM LOG: {log}")
                        self.send_insert_order(next_id, Side.SELL, self.etfs.best_bid, trade_vol, Lifespan.FAK) # sell etf
                        self.asks.add(next_id)
                        self.asks.discard(self.ask_id)
                        self.send_cancel_order(self.ask_id)
                        self.ask_id = 0
                        self.arbitrage_ids.add(next_id)
                        
    def on_order_filled_message(self, client_order_id: int, price: int, volume: int) -> None:
        """Called when when of your orders is filled, partially or fully.

        The price is the price at which the order was (partially) filled,
        which may be better than the order's limit price. The volume is
        the number of lots filled at that price.
        """
        self.logger.info("received order filled for order %d with price %d and volume %d", client_order_id,
                         price, volume)
        if client_order_id in self.arbitrage_ids:
            if client_order_id in self.bids:
                self.position += volume
                next_id = next(self.order_ids)
                log = {
                    "POSITION": self.position,
                    "FUTURES_POSITION": self.futures_position,
                    "ACTION": f"SELL {volume}x FUTURE @{MINIMUM_BID}, ID: {next_id}"
                }
                self.logger.info(f"CUSTOM LOG: {log}")
                self.send_hedge_order(next_id, Side.SELL, MINIMUM_BID, volume) # selling futures
                self.future_asks.add(next_id)
            elif client_order_id in self.asks:
                self.position -= volume
                next_id = next(self.order_ids)
                log = {
                    "POSITION": self.position,
                    "FUTURES_POSITION": self.futures_position,
                    "ACTION": f"BUY {volume}x FUTURE @{MAXIMUM_ASK//TICK_SIZE_IN_CENTS*TICK_SIZE_IN_CENTS}, ID: {next_id}"
                }
                self.logger.info(f"CUSTOM LOG: {log}")
                self.send_hedge_order(next_id, Side.BUY, MAXIMUM_ASK//TICK_SIZE_IN_CENTS*TICK_SIZE_IN_CENTS, volume) # selling futures
                self.future_bids.add(next_id)
        else:
            if client_order_id in self.bids:
                self.position += volume
                total_position = self.get_total_position()
                if total_position >= 10:
                    order_id = next(self.order_ids)
                    self.send_hedge_order(order_id, Side.ASK, MINIMUM_BID, total_position - 10) # selling futures
                    self.future_asks.add(order_id)
            elif client_order_id in self.asks:
                self.position -= volume
                total_position = self.get_total_position()
                if total_position <= -10:
                    order_id = next(self.order_ids)
                    self.send_hedge_order(order_id, Side.BID,
                                    MAXIMUM_ASK//TICK_SIZE_IN_CENTS*TICK_SIZE_IN_CENTS, 10 - total_position) # buying futures
                    self.future_bids.add(order_id)
            self.logger.info("Position: %d", self.position)

    def on_order_status_message(self, client_order_id: int, fill_volume: int, remaining_volume: int,
                                fees: int) -> None:
        """Called when the status of one of your orders changes.

        The fill_volume is the number of lots already traded, remaining_volume
        is the number of lots yet to be traded and fees is the total fees for
        this order. Remember that you pay fees for being a market taker, but
        you receive fees for being a market maker, so fees can be negative.

        If an order is cancelled its remaining volume will be zero.
        """
        self.logger.info("received order status for order %d with fill volume %d remaining %d and fees %d",
                         client_order_id, fill_volume, remaining_volume, fees)
        if remaining_volume == 0:
            if client_order_id == self.bid_id:
                self.bid_id = 0
            elif client_order_id == self.ask_id:
                self.ask_id = 0

            # It could be either a bid or an ask
            self.bids.discard(client_order_id)
            self.asks.discard(client_order_id)

    def on_trade_ticks_message(self, instrument: int, sequence_number: int, ask_prices: List[int],
                               ask_volumes: List[int], bid_prices: List[int], bid_volumes: List[int]) -> None:
        """Called periodically when there is trading activity on the market.

        The five best ask (i.e. sell) and bid (i.e. buy) prices at which there
        has been trading activity are reported along with the aggregated volume
        traded at each of those price levels.

        If there are less than five prices on a side, then zeros will appear at
        the end of both the prices and volumes arrays.
        """
        self.logger.info("received trade ticks for instrument %d with sequence number %d", instrument,
                         sequence_number)

    def one_tick_diff(self, x, y):
        if x == y:
            return x-1*TICK_SIZE_IN_CENTS, y+1*TICK_SIZE_IN_CENTS
        return x,y 

    def price(self, mid):
        if self.historical.move==0:
            return 0, 0
        if self.position>=0:
            sell_prob=0.45+self.position*0.0005
            buy_prob=0.45-self.position*0.0045
        else:
            sell_prob=0.45+self.position*0.0045
            buy_prob=0.45-self.position*0.0005
        return self.one_tick_diff(int((mid+self.historical.move*st.norm.ppf(buy_prob))//TICK_SIZE_IN_CENTS*TICK_SIZE_IN_CENTS),int((mid+self.historical.move*st.norm.ppf(1-sell_prob))//TICK_SIZE_IN_CENTS*TICK_SIZE_IN_CENTS))
