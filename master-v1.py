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
import asyncio
import itertools
import numpy as np
import scipy.stats as st

from typing import List, NamedTuple

from ready_trader_go import BaseAutoTrader, Instrument, Lifespan, MAXIMUM_ASK, MINIMUM_BID, Side


LOT_SIZE = 10
POSITION_LIMIT = 100
TICK_SIZE_IN_CENTS = 100

Order = NamedTuple('Order', [('price', int), ('vol', int), ('id', int), ('typ', str)]) # type = 'arb' or 'mm'
PotentialVolume = NamedTuple('PotentialVolume', [('max', int), ('min', int)])
Hedge = NamedTuple('Hedge', [('ETF', NamedTuple), ('FUTURE', NamedTuple)])

class OrderBook:
    def __init__(self, sequence_number: int, ask_prices: List[int], ask_volumes: List[int], bid_prices: List[int], bid_volumes: List[int]):
        self.sequence_number = sequence_number
        self.bids = [Order(price=bid_prices[i], vol=bid_volumes[i], id=-1, typ=None) for i in range(len(bid_prices))]
        self.asks = [Order(price=ask_prices[i], vol=ask_volumes[i], id=-1, typ=None) for i in range(len(ask_prices))]
        self.best_bid = self.bids[0]
        self.best_ask = self.asks[0]

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
        self.hedges = {}
        
        self.bids = {} # current
        self.future_bids = {}
        
        self.asks = {}
        self.future_asks = {}
        
        self.ask_id = self.ask_price = self.bid_id = self.bid_price = self.position = self.futures_position = 0
        
        self.historical = Historical()

    @property
    def potential_position(self):
        return PotentialVolume(max = self.position + sum([order.vol for order in self.bids.values()]), min = self.position - sum([order.vol for order in self.asks.values()]))

    @property
    def potential_fut_position(self):
        return PotentialVolume(
            max = self.futures_position + sum([order.vol for order in self.future_bids.values()]), 
            min = self.futures_position - sum([order.vol for order in self.future_asks.values()]))

    def custom_log(self, additional = {}):
        default_log = {
            "POSITION": self.position,
            "POTENTIAL_POSITION": self.potential_position,
            "FUTURES_POSITION": self.futures_position,
            "FUTURES_POTENTIAL_POSITION": self.potential_fut_position,
            "BIDS/ASKS": (self.bids, self.asks),
            "FUTURE BIDS/ASKS": (self.future_bids, self.future_asks)
        }
        default_log.update(additional)
        self.logger.info(f"CUSTOM LOG: {default_log}")

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
        if client_order_id in self.future_asks.keys(): # Bought ETF, Sold Future
            if price <= self.hedges[client_order_id].ETF.price:
                self.logger.info(f"UNFAVOURABLE HEDGE: BOUGHT ETF AT {self.hedges[client_order_id].ETF.price}, SOLD FUTURE AT {price}")
            self.futures_position -= volume
            self.future_asks.pop(client_order_id) 
        elif client_order_id in self.future_bids.keys(): # Sold ETF, Bought Future
            if price >= self.hedges[client_order_id].ETF.price:
                self.logger.info(f"UNFAVOURABLE HEDGE: SOLD ETF AT {self.hedges[client_order_id].ETF.price}, BOUGHT FUTURE AT {price}")
            self.future_bids.pop(client_order_id) 
            self.futures_position += volume
        self.hedges.pop(client_order_id)

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
        if instrument == Instrument.FUTURE: ## mm order
            order_type = 'mm'
            new_bid_price, new_ask_price = self.price(np.mean([bid_prices[0], ask_prices[0]]))
            self.custom_log({"CALCULATED PRICES": f"Bid: {new_bid_price}, Ask: {new_ask_price}"})
            if self.bid_id != 0 and new_bid_price not in (self.bid_price, 0):
                self.send_cancel_order(self.bid_id)
                self.custom_log({"CANCEL SENT": f"Bid Order:{self.bid_id}"})
                self.bid_id = 0
            if self.ask_id != 0 and new_ask_price not in (self.ask_price, 0):
                self.send_cancel_order(self.ask_id)
                self.custom_log({"CANCEL SENT": f"Ask Order:{self.ask_id}"})
                self.ask_id = 0
            if self.bid_id == 0 and new_bid_price != 0 and self.potential_position.max+LOT_SIZE <= POSITION_LIMIT:
                self.bid_id = next(self.order_ids)
                self.bid_price = new_bid_price
                self.send_insert_order(self.bid_id, Side.BUY, new_bid_price, LOT_SIZE, Lifespan.GOOD_FOR_DAY)
                self.bids[self.bid_id] = Order(price=new_bid_price, vol=LOT_SIZE, id=self.bid_id, typ=order_type)
                self.custom_log({"ACTION": f"BUY {LOT_SIZE} ETF @{new_bid_price}, ID: {self.bid_id}, type: {order_type}"})
            if self.ask_id == 0 and new_ask_price != 0 and self.potential_position.min-LOT_SIZE >= -POSITION_LIMIT:
                self.ask_id = next(self.order_ids)
                self.ask_price = new_ask_price
                self.send_insert_order(self.ask_id, Side.SELL, new_ask_price, LOT_SIZE, Lifespan.GOOD_FOR_DAY)
                self.asks[self.ask_id] = Order(price=new_ask_price, vol=LOT_SIZE, id=self.ask_id, typ=order_type)
                self.custom_log({"ACTION": f"SELL {LOT_SIZE} ETF @{new_ask_price}, ID: {self.ask_id}, type: {order_type}"})
            # Load Order book into memory
            self.futures = OrderBook(sequence_number, ask_prices,ask_volumes, bid_prices, bid_volumes)   
        
        elif instrument == Instrument.ETF: ## arb order
            order_type = 'arb'
            # Load Order book into memory
            self.etfs = OrderBook(sequence_number, ask_prices, ask_volumes, bid_prices, bid_volumes)
            ## ARBITRAGE:
            if self.etfs.best_ask.price < self.futures.best_bid.price: # etf buy, future sell opp.
                # All favourable etf ask prices to trade with
                arb_asks = [ask for ask in self.etfs.asks if ask.price < self.futures.best_bid.price]
                # Keeping potential counts so that we can track our position before official fills
                potential_fut_vol = self.futures.best_bid.vol
                for etf_ask in arb_asks:
                    trade_vol = min(
                        POSITION_LIMIT-self.potential_position.max, 
                        etf_ask.vol, 
                        potential_fut_vol, 
                        POSITION_LIMIT+self.potential_fut_position.min)
                    if trade_vol > 0:
                        potential_fut_vol -= trade_vol
                        next_id = next(self.order_ids)
                        self.send_insert_order(client_order_id=next_id, side=Side.BUY, price=etf_ask.price, volume=trade_vol, lifespan=Lifespan.FAK) # buy etf
                        self.bids[next_id] = Order(price=etf_ask.price, vol=trade_vol, id=next_id, typ=order_type)
                        self.custom_log({"ACTION": f"BUY {trade_vol} ETF @{etf_ask.price}, ID: {next_id}, type: {order_type}"})

            if self.futures.best_ask.price < self.etfs.best_bid.price: # future buy, etf sell opp.
                arb_bids = [bid for bid in self.etfs.bids if bid.price > self.futures.best_ask.price]
                potential_fut_vol = self.futures.best_bid.vol
                for etf_bid in arb_bids:
                    trade_vol = min(
                        POSITION_LIMIT+self.potential_position.min, 
                        etf_bid.vol, 
                        potential_fut_vol, 
                        POSITION_LIMIT-self.potential_fut_position.max)
                    if trade_vol > 0:
                        potential_fut_vol -= trade_vol
                        next_id = next(self.order_ids)
                        self.send_insert_order(client_order_id=next_id, side=Side.SELL, price=etf_bid.price, volume=trade_vol, lifespan=Lifespan.FAK) # sell etf
                        self.asks[next_id] = Order(price=etf_bid.price, vol=trade_vol, id=next_id, typ=order_type)
                        self.custom_log({"ACTION": f"SELL {trade_vol} ETF @{etf_bid.price}, ID: {next_id}, type: {order_type}"})
                        
    def on_order_filled_message(self, client_order_id: int, price: int, volume: int) -> None:
        """Called when when of your orders is filled, partially or fully.

        The price is the price at which the order was (partially) filled,
        which may be better than the order's limit price. The volume is
        the number of lots filled at that price.
        """
        self.logger.info("received order filled for order %d with price %d and volume %d", client_order_id,
                         price, volume)
        if client_order_id in self.bids.keys():
            self.position += volume
            current_bid = self.bids[client_order_id]
            
            if current_bid.typ == 'arb': # if arb complete hedge
                next_id = next(self.order_ids)
                trade_vol = volume
                self.send_hedge_order(client_order_id=next_id, side=Side.SELL, price=MINIMUM_BID, volume=trade_vol) # selling futures (mkt order)
                self.bids[client_order_id] = Order(id=client_order_id, price=price, vol=(self.bids[client_order_id].vol-volume), typ=current_bid.typ)
                self.future_asks[next_id] = Order(id=next_id, price=MINIMUM_BID, vol=trade_vol, typ='arb')
                self.custom_log({"ACTION": f"SELL {trade_vol}x FUTURE @{MINIMUM_BID}, ID: {next_id}, type: {current_bid.typ}"})
                self.hedges[next_id] = Hedge(ETF = current_bid, FUTURE = Order(price=None, vol=volume, id=next_id, typ='arb'))
                
        elif client_order_id in self.asks.keys():
            self.position -= volume
            current_ask = self.asks[client_order_id]

            if current_ask.typ == 'arb': # if arb complete hedge
                next_id = next(self.order_ids)
                trade_vol = volume
                self.send_hedge_order(client_order_id=next_id, side=Side.SELL, price=MAXIMUM_ASK//TICK_SIZE_IN_CENTS*TICK_SIZE_IN_CENTS, volume=trade_vol) # buying futures (mkt order)
                self.asks[client_order_id] = Order(id=client_order_id, price=price, vol=(self.asks[client_order_id].vol-volume), typ=current_ask.typ) # ask etf order update
                self.future_bids[next_id] = Order(id=next_id, price=MAXIMUM_ASK//TICK_SIZE_IN_CENTS*TICK_SIZE_IN_CENTS, vol=trade_vol, typ='arb')
                self.custom_log({"ACTION": f"BUY {volume}x FUTURE @{MAXIMUM_ASK//TICK_SIZE_IN_CENTS*TICK_SIZE_IN_CENTS}, ID: {next_id}, type: {current_ask.typ}"})
                self.hedges[next_id] = Hedge(ETF = current_ask, FUTURE = Order(price=None, vol=volume, id=next_id, typ='arb'))

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
            if client_order_id in self.bids.keys():
                self.bids.pop(client_order_id)
            elif client_order_id in self.asks.keys():
                self.asks.pop(client_order_id)            
            self.custom_log()

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
        if self.position==100:
            bid=0
        else:
            bid=max(int((mid+self.historical.move*st.norm.ppf(buy_prob))//TICK_SIZE_IN_CENTS*TICK_SIZE_IN_CENTS),0)
        if self.position==-100:
            ask=0
        else:
            ask=int((mid+self.historical.move*st.norm.ppf(1-sell_prob)))//TICK_SIZE_IN_CENTS*TICK_SIZE_IN_CENTS

        return self.one_tick_diff(bid,ask)
