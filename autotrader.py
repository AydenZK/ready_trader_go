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

from typing import List

from ready_trader_go import BaseAutoTrader, Instrument, Lifespan, MAXIMUM_ASK, MINIMUM_BID, Side


LOT_SIZE = 10
POSITION_LIMIT = 100
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


    def update(self, instrument, price):
        self.historical_prices[instrument].append(price)

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
            self.future_asks.discard(client_order_id)
        if client_order_id in self.future_bids:
            self.futures_position += volume
            self.future_bids.discard(client_order_id)

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
        # self.logger.info(f"TIME: {self.event_loop.time()}")
        # adds to historical prices (mid price)
        self.historical.update(instrument, np.mean([bid_prices[0], ask_prices[0]]))
        if instrument == Instrument.FUTURE:
            self.futures = OrderBook(sequence_number, ask_prices,ask_volumes, bid_prices, bid_volumes)
        else:
            self.etfs = OrderBook(sequence_number, ask_prices, ask_volumes, bid_prices, bid_volumes)
    
        if instrument == Instrument.ETF:
            if self.futures.best_bid > self.etfs.best_bid:
                if 0 < self.etfs.best_ask_vol <= self.futures.best_bid_vol and self.etfs.best_ask < self.futures.best_bid:
                    trade_vol = min(POSITION_LIMIT-self.position, self.etfs.best_ask_vol)
                    if trade_vol > 0:
                        next_id = next(self.order_ids)
                        log = {
                            "POSITION": self.position,
                            "MAX_VOLUME": self.etfs.best_bid_vol,
                            "ACTION": f"BUY {trade_vol} ETF @{self.etfs.best_ask}, ID: {next_id}"
                        }
                        self.logger.info(f"CUSTOM LOG: {log}")
                        self.send_insert_order(next_id, Side.BUY, self.etfs.best_ask, trade_vol, Lifespan.FAK) # buy etf
                        self.bids.add(next_id)

            if self.etfs.best_bid > self.futures.best_bid:
                if 0 < self.etfs.best_bid_vol <= self.futures.best_ask_vol and self.futures.best_ask < self.etfs.best_bid:
                    trade_vol = min(POSITION_LIMIT+self.position, self.etfs.best_bid_vol)
                    if trade_vol > 0:
                        next_id = next(self.order_ids)
                        log = {
                            "POSITION": self.position,
                            "MAX_VOLUME": self.etfs.best_bid_vol,
                            "ACTION": f"SELL {trade_vol} ETF @{self.etfs.best_bid}, ID: {next_id}"
                        }
                        self.logger.info(f"CUSTOM LOG: {log}")
                        self.send_insert_order(next_id, Side.SELL, self.etfs.best_bid, trade_vol, Lifespan.FAK) # sell etf
                        self.asks.add(next_id)
                        
    def on_order_filled_message(self, client_order_id: int, price: int, volume: int) -> None:
        """Called when when of your orders is filled, partially or fully.

        The price is the price at which the order was (partially) filled,
        which may be better than the order's limit price. The volume is
        the number of lots filled at that price.
        """
        self.logger.info("received order filled for order %d with price %d and volume %d", client_order_id,
                         price, volume)
        if client_order_id in self.bids:
            self.position += volume
            self.send_hedge_order(next(self.order_ids), Side.SELL, MINIMUM_BID, volume) # selling futures
        elif client_order_id in self.asks:
            self.position -= volume
            self.send_hedge_order(next(self.order_ids), Side.BUY, MAXIMUM_ASK//TICK_SIZE_IN_CENTS*TICK_SIZE_IN_CENTS, volume) # selling futures

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
