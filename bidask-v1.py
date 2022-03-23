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
from turtle import position
import numpy as np
import scipy.stats as st

from typing import List, NamedTuple

from ready_trader_go import BaseAutoTrader, Instrument, Lifespan, MAXIMUM_ASK, MINIMUM_BID, Side


LOT_SIZE = 10
POSITION_LIMIT = 100
TICK_SIZE_IN_CENTS = 100

Order = NamedTuple('Order', [('price', int), ('vol', int), ('id', int), ('edge', float)])
PotentialVolume = NamedTuple('PotentialVolume', [('max', int), ('min', int)])

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
        self.bids = {}
        self.future_bids = set()
        self.asks = {}
        self.future_asks = set()
        self.ask_id = self.ask_price = self.bid_id = self.bid_price = self.position = self.futures_position = 0
        self.historical = Historical()
        self.dollar_edge = 0

    @property
    def potential_position(self):
        return PotentialVolume(max = self.position + sum([order.vol for order in self.bids.values()]), min = self.position - sum([order.vol for order in self.asks.values()]))

    @property
    def total_position(self):
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
            pass
        if client_order_id in self.future_bids:
            pass
            
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
        theo: float = self.get_theo(bid_prices, bid_volumes, ask_prices, ask_volumes)
        self.historical.update(instrument, theo)
        
        if instrument == Instrument.FUTURE:
            edges: tuple[float, float] = self.get_edges(theo, position=self.position)
            volumes: tuple[int, int] = self.get_volumes(position=self.position)
            bid_order, ask_order = self.get_orders(theo, edges, volumes)

            self.logger.info(f"Calculated Bid/Ask: Bid: {bid_order}, Ask: {ask_order}" )
            
            if self.bid_id != 0 and bid_order and bid_order.price not in (self.bid_price, 0):
                self.send_cancel_order(self.bid_id)
                self.logger.info("Cancel sent for order %d", self.bid_id)
                self.bid_id = 0
            if self.ask_id != 0 and ask_order and ask_order.price not in (self.ask_price, 0):
                self.send_cancel_order(self.ask_id)
                self.logger.info("Cancel sent for order %d", self.ask_id)
                self.ask_id = 0

            if bid_order and self.bid_id == 0 and bid_order.price != 0 and self.potential_position.max+bid_order.vol <= POSITION_LIMIT:
                self.bid_id = next(self.order_ids)
                self.bid_price = bid_order.price
                self.send_insert_order(self.bid_id, Side.BUY, bid_order.price, bid_order.vol, Lifespan.GOOD_FOR_DAY)
                self.bids[self.bid_id] = Order(price=bid_order.price, vol=bid_order.vol, id=self.bid_id, edge=bid_order.edge)

            if ask_order and self.ask_id == 0 and ask_order.price != 0 and self.potential_position.min-ask_order.vol >= -POSITION_LIMIT:
                self.ask_id = next(self.order_ids)
                self.ask_price = ask_order.price
                self.send_insert_order(self.ask_id, Side.SELL, ask_order.price, ask_order.vol, Lifespan.GOOD_FOR_DAY)
                self.asks[self.ask_id] = Order(price=ask_order.price, vol=ask_order.vol, id=self.ask_id, edge=bid_order.edge)

            self.logger.info(f"Bids: {self.bids}, Asks {self.asks}")
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
            self.bids[client_order_id] = Order(id=client_order_id, price=price, vol=(self.bids[client_order_id].vol-volume), edge=self.bids[client_order_id].edge)
            self.dollar_edge += volume * self.bids[client_order_id].edge
            total_position = self.total_position
            if total_position > 10:
                order_id = next(self.order_ids)
                self.send_hedge_order(order_id, Side.ASK, MINIMUM_BID, total_position-10) # selling futures
                self.future_asks.add(order_id)
                self.futures_position -= (total_position-10)
        elif client_order_id in self.asks.keys():
            self.position -= volume
            self.asks[client_order_id] = Order(id=client_order_id, price=price, vol=(self.asks[client_order_id].vol-volume), edge=self.asks[client_order_id].edge)
            self.dollar_edge += volume * self.asks[client_order_id].edge
            total_position = self.total_position
            if total_position < -10:
                order_id = next(self.order_ids)
                self.send_hedge_order(order_id, Side.BID,
                                  MAXIMUM_ASK//TICK_SIZE_IN_CENTS*TICK_SIZE_IN_CENTS, -total_position-10) # buying futures
                self.future_bids.add(order_id)
                self.futures_position += (-total_position-10)
        self.logger.info(f"Position: {self.position}, TOTAL_EDGE: {self.dollar_edge}")
        
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
            self.logger.info(f"Bids: {self.bids}, Asks {self.asks}")
            
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

    def get_theo(self, bid_prices, bid_volumes, ask_prices, ask_volumes) -> float:
        """Get theoretical price"""
        return np.mean([bid_prices[0], ask_prices[0]])

    def get_edges(self, theo, position) -> tuple[int, int]:
        """Return bid and ask edge from theo"""
        if position>=0:
            sell_prob=0.45+position*0.0005
            buy_prob=0.45-position*0.0045

        else:
            sell_prob=0.45+position*0.0045
            buy_prob=0.45-position*0.0005

        bid_edge = self.historical.move*st.norm.ppf(buy_prob)
        ask_edge = self.historical.move*st.norm.ppf(1-sell_prob)

        return bid_edge, ask_edge

    def get_volumes(self, position) -> tuple[int, int]:
        """Return bid and ask volumes"""
        return (LOT_SIZE, LOT_SIZE)

    def one_tick_diff(self, x, y):
        if x and y and x.price == y.price:
            return (
                Order(x.price-TICK_SIZE_IN_CENTS, x.vol, x.id, x.edge+TICK_SIZE_IN_CENTS),
                Order(y.price+TICK_SIZE_IN_CENTS, y.vol, y.id, y.edge+TICK_SIZE_IN_CENTS)
            )
        return x,y 

    def get_orders(self, theo, edges, volumes) -> tuple[Order, Order]:
        bid = None
        ask = None
        if self.historical.move==0:
            return (None, None)

        if self.position != 100:
            bid_price = int(round((theo + edges[0])/TICK_SIZE_IN_CENTS)*TICK_SIZE_IN_CENTS)
            if bid_price > 0:
                bid_vol = volumes[0]
                bid = Order(bid_price, bid_vol, -1, abs(theo-bid_price))
        
        if self.position != -100:
            ask_price = int(round((theo + edges[1])/TICK_SIZE_IN_CENTS)*TICK_SIZE_IN_CENTS)
            ask_vol = volumes[1]
            ask = Order(ask_price, ask_vol, -1, abs(theo-ask_price))

        return self.one_tick_diff(bid, ask)

