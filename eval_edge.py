import pandas as pd

def evaluate_theo(trader_log, score_board, trader, details=''):
    with open(trader_log, 'r') as f:
        total_edge = 0
        for line in f.readlines():
            if 'TOTAL_EDGE' in line:
                total_edge = float(line.split('TOTAL_EDGE: ')[-1])

        df = pd.read_csv(score_board)
        res = df[df['Team']==trader].tail(1)
        trade_profit = res['ProfitOrLoss'] - (res['EtfPosition'] * res['EtfPrice'] + res['FuturePosition'] * res['FuturePrice']) + res['TotalFees']
        true_edge = trade_profit.values[0]

        pct_error = f"{round((-true_edge + total_edge) / true_edge * 10000) / 100} %"
        abs_error = abs(true_edge - total_edge)
        details += " vs " + "|".join(df['Team'].unique())

        with open('data/theo_eval.csv', 'a') as res:
            res.write(f'\n{trader},{details},{total_edge},{true_edge},{pct_error},{abs_error}')


trader_log = 'bidask-v1.log' # log
score_board = 'score_board.csv' # log

evaluate_theo(trader_log=trader_log, score_board=score_board, trader="BidAsk-v1", details='md3_x3_4min')