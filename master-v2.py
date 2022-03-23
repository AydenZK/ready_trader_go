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
ARB_LIMIT = 95
TICK_SIZE_IN_CENTS = 100
HEDGE_TIME_LIMIT = 14 # secs

Order = NamedTuple('Order', [('price', int), ('vol', int), ('id', int), ('typ', str), ('edge', float)])
PotentialVolume = NamedTuple('PotentialVolume', [('max', int), ('min', int)])
Hedge = NamedTuple('Hedge', [('ETF', NamedTuple), ('FUTURE', NamedTuple)])

EMPTY_ORDER = Order(price=0, vol=0, id=-1, typ='empty', edge=0)

class UnhedgedTimer:
    def __init__(self, event_loop):
        self.event_loop = event_loop
        self.start_time = None
        self.is_active = False
    
    @property
    def seconds(self):
        return self.event_loop.time() - self.start_time if self.is_active else 0

    def start(self):
        self.is_active = True
        self.start_time = self.event_loop.time()

    def reset(self):
        self.is_active = False
        self.start_time = None

class OrderBook:
    def __init__(self, sequence_number: int, ask_prices: List[int], ask_volumes: List[int], bid_prices: List[int], bid_volumes: List[int]):
        self.sequence_number = sequence_number
        self.bids = [Order(price=bid_prices[i], vol=bid_volumes[i], id=-1, typ=None, edge=0) for i in range(len(bid_prices))]
        self.asks = [Order(price=ask_prices[i], vol=ask_volumes[i], id=-1, typ=None, edge=0) for i in range(len(ask_prices))]
        self.best_bid = self.bids[0]
        self.best_ask = self.asks[0]

class Historical:
    def __init__(self):
        self.l2_etf = []
        self.move=0
        self.total=0
        self.n=0
        
    def update(self, price):
        self.l2_etf.append(price)
        if len(self.l2_etf) == 3:
            self.l2_etf.pop(0)
            self.n+=1
            self.total+=(self.l2_etf[-1]-self.l2_etf[-2])**2
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
        self.canceled_ids = set()
        self.hedges = {}
        self.unhedged_timer = UnhedgedTimer(event_loop=self.event_loop)

        self.safe_to_trade = True

        self.bids = {} # current
        self.future_bids = {}
        
        self.asks = {}
        self.future_asks = {}
        
        self.ask_id = self.ask_price = self.bid_id = self.bid_price = self.position = self.futures_position = 0
        
        self.historical = Historical()

    @property
    def potential_position(self):
        return PotentialVolume(
            max = self.position + sum([order.vol for order in self.bids.values()]), 
            min = self.position - sum([order.vol for order in self.asks.values()]))

    @property
    def potential_fut_position(self):
        return PotentialVolume(
            max = self.futures_position + sum([order.vol for order in self.future_bids.values()]), 
            min = self.futures_position - sum([order.vol for order in self.future_asks.values()]))

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

        if not self.safe_to_trade and self.future_asks.get(client_order_id, EMPTY_ORDER).typ == 'safe_hedge' or self.future_bids.get(client_order_id, EMPTY_ORDER).typ == 'safe_hedge':
            self.safe_to_trade = True
        
        if client_order_id in self.future_asks.keys(): # Bought ETF, Sold Future
            self.futures_position -= volume
            self.future_asks.pop(client_order_id) 

        elif client_order_id in self.future_bids.keys(): # Sold ETF, Bought Future
            self.future_bids.pop(client_order_id) 
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
        theo = self.get_theo(bid_prices, bid_volumes, ask_prices, ask_volumes)
                
        if instrument == Instrument.FUTURE and self.safe_to_trade: ## mm order:
            edges = self.get_edges(theo, position=self.position)
            volumes = self.get_volumes(position=self.position)
            bid_order, ask_order = self.get_orders(theo, edges, volumes)

            self.logger.info(f"Calculated Bid/Ask: Bid: {bid_order}, Ask: {ask_order}" )
            
            order_type = 'mm'
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
                self.bids[self.bid_id] = Order(price=bid_order.price, vol=bid_order.vol, id=self.bid_id, typ='mm', edge=bid_order.edge)

            if ask_order and self.ask_id == 0 and ask_order.price != 0 and self.potential_position.min-ask_order.vol >= -POSITION_LIMIT:
                self.ask_id = next(self.order_ids)
                self.ask_price = ask_order.price
                self.send_insert_order(self.ask_id, Side.SELL, ask_order.price, ask_order.vol, Lifespan.GOOD_FOR_DAY)
                self.asks[self.ask_id] = Order(price=ask_order.price, vol=ask_order.vol, id=self.ask_id, typ='mm', edge=ask_order.edge)

            self.futures = OrderBook(sequence_number, ask_prices,ask_volumes, bid_prices, bid_volumes)   
        
        elif instrument == Instrument.ETF and self.safe_to_trade: ## arb order
            self.historical.update(theo)
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
                        ARB_LIMIT-self.potential_position.max, 
                        etf_ask.vol, 
                        potential_fut_vol, 
                        ARB_LIMIT+self.potential_fut_position.min)
                    if trade_vol > 0:
                        potential_fut_vol -= trade_vol
                        next_id = next(self.order_ids)
                        self.send_insert_order(client_order_id=next_id, side=Side.BUY, price=etf_ask.price, volume=trade_vol, lifespan=Lifespan.FAK) # buy etf
                        self.bids[next_id] = Order(price=etf_ask.price, vol=trade_vol, id=next_id, typ=order_type, edge=0)
                        self.future_asks[next_id+0.5] = Order(price=None, vol=trade_vol, id=None, typ='arb', edge=0) # potential

            if self.futures.best_ask.price < self.etfs.best_bid.price: # future buy, etf sell opp.
                arb_bids = [bid for bid in self.etfs.bids if bid.price > self.futures.best_ask.price]
                potential_fut_vol = self.futures.best_bid.vol
                for etf_bid in arb_bids:
                    trade_vol = min(
                        ARB_LIMIT+self.potential_position.min, 
                        etf_bid.vol, 
                        potential_fut_vol, 
                        ARB_LIMIT-self.potential_fut_position.max)
                    if trade_vol > 0:
                        potential_fut_vol -= trade_vol
                        next_id = next(self.order_ids)
                        self.send_insert_order(client_order_id=next_id, side=Side.SELL, price=etf_bid.price, volume=trade_vol, lifespan=Lifespan.FAK) # sell etf
                        self.asks[next_id] = Order(price=etf_bid.price, vol=trade_vol, id=next_id, typ=order_type, edge=0)
                        self.future_bids[next_id+0.5] = Order(price=None, vol=trade_vol, id=None, typ='arb', edge=0) 
            
            unhedged = abs(self.futures_position + self.position) > 10
            if unhedged:
                if not self.unhedged_timer.is_active:
                    self.unhedged_timer.start()
                elif self.unhedged_timer.seconds > HEDGE_TIME_LIMIT:
                    self.safe_to_trade = False
                    self.canceled_ids = self.cancel_all_orders()
            else: # we are hedged
                if self.unhedged_timer.is_active:
                    self.unhedged_timer.reset()

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
            self.bids[client_order_id] = Order(id=client_order_id, price=price, vol=(self.bids[client_order_id].vol-volume), typ=self.bids[client_order_id].typ, edge=self.bids[client_order_id].edge)
            
            if current_bid.typ == 'arb': # if arb complete hedge
                next_id = next(self.order_ids)
                trade_vol = volume
                self.send_hedge_order(client_order_id=next_id, side=Side.SELL, price=MINIMUM_BID, volume=trade_vol) # selling futures (mkt order)
                self.future_asks.pop(client_order_id+0.5)
                self.future_asks[next_id] = Order(id=next_id, price=MINIMUM_BID, vol=trade_vol, typ=current_bid.typ, edge=0)
                self.hedges[next_id] = Hedge(ETF = current_bid, FUTURE = Order(price=None, vol=trade_vol, id=None, typ='arb', edge=0))
                
        elif client_order_id in self.asks.keys():
            self.position -= volume
            current_ask = self.asks[client_order_id]
            self.asks[client_order_id] = Order(id=client_order_id, price=price, vol=(self.asks[client_order_id].vol-volume),typ= self.asks[client_order_id].typ, edge=self.asks[client_order_id].edge)

            if current_ask.typ == 'arb': # if arb complete hedge
                next_id = next(self.order_ids)
                trade_vol = volume
                self.send_hedge_order(client_order_id=next_id, side=Side.BUY, price=MAXIMUM_ASK//TICK_SIZE_IN_CENTS*TICK_SIZE_IN_CENTS, volume=trade_vol) # buying futures (mkt order)
                self.future_bids.pop(client_order_id+0.5)
                self.future_bids[next_id] = Order(id=next_id, price=MAXIMUM_ASK//TICK_SIZE_IN_CENTS*TICK_SIZE_IN_CENTS, vol=trade_vol, typ=current_ask.typ, edge=0)
                self.hedges[next_id] = Hedge(ETF = current_ask, FUTURE = Order(price=None, vol=volume, id=next_id, typ='arb', edge=0))

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

            if client_order_id in self.canceled_ids and not self.safe_to_trade:
                self.canceled_ids.discard(client_order_id)
                if len(self.bids) == len(self.asks) == len(self.canceled_ids) == 0:
                    next_id = next(self.order_ids)
                    if self.position + self.futures_position > 10:
                        trade_vol = self.position + self.futures_position
                        self.send_hedge_order(client_order_id=next_id, side=Side.SELL, price=MINIMUM_BID, volume=trade_vol) # selling futures (mkt order)
                        self.future_asks[next_id] = Order(id=next_id, price=MINIMUM_BID, vol=trade_vol, typ='safe_hedge', edge=0)
                    
                    elif self.position + self.futures_position < -10:
                        trade_vol = -self.position - self.futures_position
                        self.send_hedge_order(client_order_id=next_id, side=Side.BUY, price=MAXIMUM_ASK//TICK_SIZE_IN_CENTS*TICK_SIZE_IN_CENTS, volume=trade_vol) # buying futures (mkt order)
                        self.future_bids[next_id] = Order(id=next_id, price=MAXIMUM_ASK//TICK_SIZE_IN_CENTS*TICK_SIZE_IN_CENTS, vol=trade_vol, typ='safe_hedge', edge=0)


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
        if bid_volumes[0]+ask_volumes[0]>0:
            num=sum([(bid_prices[i]*bid_volumes[i]+ask_prices[i]*ask_volumes[i])*np.exp(-i) for i in range(len(bid_volumes))])
            denom=sum([(bid_volumes[i]+ask_volumes[i])*np.exp(-i) for i in range(len(bid_volumes))])
            return (num/denom)
        else:
            return np.mean([bid_prices[0],ask_prices[0]])

    def get_edges(self, theo, position):
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

    def get_volumes(self, position):
        """Return bid and ask volumes"""
        return (LOT_SIZE, LOT_SIZE)

    def one_tick_diff(self, x, y):
        if x and y and x.price == y.price:
            return (
                Order(x.price-TICK_SIZE_IN_CENTS, x.vol, x.id, None, x.edge+TICK_SIZE_IN_CENTS),
                Order(y.price+TICK_SIZE_IN_CENTS, y.vol, y.id, None, y.edge+TICK_SIZE_IN_CENTS)
            )
        return x,y 

    def get_orders(self, theo, edges, volumes):
        bid = None
        ask = None
        if self.historical.move==0:
            return (None, None)

        if self.position != 100:
            bid_price = int(round((theo + edges[0])/TICK_SIZE_IN_CENTS)*TICK_SIZE_IN_CENTS)
            if bid_price > 0:
                bid_vol = volumes[0]
                bid = Order(bid_price, bid_vol, -1, None, abs(theo-bid_price))
        
        if self.position != -100:
            ask_price = int(round((theo + edges[1])/TICK_SIZE_IN_CENTS)*TICK_SIZE_IN_CENTS)
            ask_vol = volumes[1]
            ask = Order(ask_price, ask_vol, -1, None, abs(theo-ask_price))

        return self.one_tick_diff(bid, ask)

    def cancel_all_orders(self):
        """Cancel all current orders"""
        cancelled_ids = set()
        for order_id in self.asks.keys():
            self.send_cancel_order(order_id)
            cancelled_ids.add(order_id)
        for order_id in self.bids.keys():
            self.send_cancel_order(order_id)
            cancelled_ids.add(order_id)
        return cancelled_ids
