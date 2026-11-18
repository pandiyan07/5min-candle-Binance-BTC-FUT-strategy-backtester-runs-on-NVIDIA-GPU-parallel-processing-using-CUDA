import pandas as pd
import numpy as np
import pandas_ta as ta
import cupy as cp
from joblib import Memory
from datetime import datetime

# Set up caching for technical indicator calculations
memory = Memory("./cache", verbose=0)

def CALCULATE_EMA(df, length):
    df['oc2'] = (df['open'].astype(float) + df['close'].astype(float)) / 2
    return ta.ema(df['oc2'], length=length)

def CALCULATE_RSI(df, rsi_length):
    return ta.rsi(df['ema_34'], length=rsi_length)

# GPU Kernel for Backtesting
BACKTEST_KERNEL = cp.RawKernel(
    r'''
    extern "C" __global__
    void BACKTEST_KERNEL(
        const float* rsi_14, const float* close_price,
        const float* ema_5, const float* ema_34,
        const float* params,
        float* results, 
        const int num_candles,
        const int num_params)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= num_params) return;

        float ema34_LONG_ENTRY_variation = params[tid * 3];
        float ema34_SHORT_ENTRY_variation = params[tid * 3 + 1];
        float TSL_value = params[tid * 3 + 2];

        int result_idx = tid * 14;
        float long_total_profit = 0.0;
        float long_total_loss = 0.0;
        float short_total_profit = 0.0;
        float short_total_loss = 0.0;
        int num_long_trades = 0;
        int num_short_trades = 0;
        float position = 0.0;
        float entry_price = 0.0;
        float highest_price = 0.0;
        float lowest_price = 0.0;

        for (int j = 2; j < num_candles; j++) {
            float close = close_price[j];
            float rsi =  rsi_14[j];

            bool long_condition = (ema_5[j] > ema_34[j]) && ((ema_34[j] - ema_34[j-4]) > ema34_LONG_ENTRY_variation) && (rsi > 70);
            bool short_condition = (ema_5[j] < ema_34[j]) && ((ema_34[j-4] - ema_34[j]) > ema34_SHORT_ENTRY_variation) && (rsi < 30);

            if (position == 0.0) {
                if (long_condition) {
                    position = 1.0;
                    entry_price = close;
                    highest_price = close;  // Track the highest price for trailing stop
                    num_long_trades++;
                } else if (short_condition) {
                    position = 2.0;
                    entry_price = close;
                    lowest_price = close;  // Track the lowest price for trailing stop
                    num_short_trades++;
                }
            } else if (position == 1.0) {
                highest_price = fmaxf(highest_price, close);  // Update highest price
                bool ema_exit = ema_5[j] < ema_34[j];
                bool trailing_stop_loss = (close < (highest_price - TSL_value));

                if (ema_exit || trailing_stop_loss) {
                    float profit = close - entry_price;
                    (profit > 0) ? long_total_profit += profit : long_total_loss += profit; 
                    position = 0.0;
                }
            } else if (position == 2.0) {
                lowest_price = fminf(lowest_price, close);  // Update lowest price
                bool ema_exit = ema_5[j] > ema_34[j];
                bool trailing_stop_loss = (close > (lowest_price + TSL_value));

                if (ema_exit || trailing_stop_loss) {
                    float profit = entry_price - close;
                    (profit > 0) ? short_total_profit += profit : short_total_loss += profit;
                    position = 0.0;
                }
            }

            // Force close the last open trade if it exists on the last candle
            if (j == num_candles - 1 && position != 0.0) {  
                float profit = (position == 1.0) ? (close - entry_price) : (entry_price - close);
                (profit > 0) ? long_total_profit += profit : long_total_loss += profit;
                position = 0.0;  // Reset position after closing
            }
        }

        results[result_idx] = ema34_LONG_ENTRY_variation;
        results[result_idx + 1] = ema34_SHORT_ENTRY_variation; 
        results[result_idx + 2] = TSL_value;
        results[result_idx + 3] = num_long_trades + num_short_trades;
        results[result_idx + 4] = num_long_trades;
        results[result_idx + 5] = num_short_trades;
        results[result_idx + 6] = (num_long_trades + num_short_trades > 0) ? ((long_total_profit + short_total_profit) / ((long_total_profit + short_total_profit) + fabsf(long_total_loss + short_total_loss))) * 100.0f : 0.0f;
        results[result_idx + 7] = long_total_profit;
        results[result_idx + 8] = long_total_loss;
        if ((long_total_profit) > fabsf(long_total_loss)) {
            results[result_idx + 9] = (fabsf(long_total_loss) / (long_total_profit)) * 100.0f;
        } else {
            results[result_idx + 9] = 0.0f;
        }
        results[result_idx + 10] = short_total_profit;
        results[result_idx + 11] = short_total_loss;
        if ((short_total_profit) > fabsf(short_total_loss)) {
            results[result_idx + 12] = (fabsf(short_total_loss) / (short_total_profit)) * 100.0f;
        } else {
            results[result_idx + 12] = 0.0f;
        }
        results[result_idx + 13] = (long_total_profit + short_total_profit) - fabsf(long_total_loss + short_total_loss); 
    }
        ''',
    'BACKTEST_KERNEL'
)

def BACKTEST_GPU(df, params):
    num_candles = len(df)
    num_params = len(params)
    results = cp.zeros((num_params * 14), dtype=cp.float32)
    threads_per_block = 256
    blocks_per_grid = (num_params + threads_per_block - 1) // threads_per_block

    gpu_data = {
        'rsi_14': cp.asarray(df['rsi_14'], dtype=cp.float32),
        'close': cp.asarray(df['close'], dtype=cp.float32),
        'ema_5': cp.asarray(df['ema_5'], dtype=cp.float32),
        'ema_34': cp.asarray(df['ema_34'], dtype=cp.float32),
        'params': cp.asarray(params, dtype=cp.float32)
    }
    
    BACKTEST_KERNEL((blocks_per_grid,), (threads_per_block,), (
        gpu_data['rsi_14'],
        gpu_data['close'],
        gpu_data['ema_5'],
        gpu_data['ema_34'],
        gpu_data['params'],
        results,
        num_candles,
        num_params
    ))
    return results.get().reshape((num_params, 14))

def TRADING_BOT(df):
    df['ema_5'] = CALCULATE_EMA(df, 5)
    df['ema_34'] = CALCULATE_EMA(df, 34)
    df['rsi_14'] = CALCULATE_RSI(df, 14)
    
    params = np.array([
        (ema34_LONG_ENTRY_variation, ema34_SHORT_ENTRY_variation, TSL_value)
        for ema34_LONG_ENTRY_variation in range(50, 300, 50)
        for ema34_SHORT_ENTRY_variation in range(50, 300, 50)
        for TSL_value in range(0, 550, 50)
    ], dtype=np.float32)
    
    start_time = datetime.now()
    results = BACKTEST_GPU(df, params)
    end_time = datetime.now()
    
    print(f'Backtesting completed in {end_time - start_time} seconds')
    df_results = pd.DataFrame(results, columns=[
        'EMA34 Long ENTRY variation', 
        'EMA34 Short ENTRY variation',
        'TSL value',
        'Total Trades', 
        'Long Trades', 
        'Short Trades', 
        'Win %', 
        'Long Total Profit', 
        'Long Total Loss',
        'Long Loss perc from Profit',
        'Short Total Profit', 
        'Short Total Loss', 
        'Short Loss perc from Profit',
        'Net Profit',
        ])
    return df_results


if __name__ == "__main__":
    INPUT_FILE = r"5min ohlc MARCH.csv"
    OUTPUT_FILE_NET_PROFIT = r"5min ohlc MARCH v7.csv"
    
    df = pd.read_csv(INPUT_FILE)
    output_df = TRADING_BOT(df)
    
    # Sort by net profit (descending) to see the best-performing parameter combinations.
    output_df_net_profit = output_df.sort_values('Net Profit', ascending=False)
    output_df_net_profit.to_csv(OUTPUT_FILE_NET_PROFIT, index=False)
    
    table_generation_end_time = datetime.now()
    print('\nTop Highest net profit =')
    print(output_df_net_profit)
    print('\nTrading Bot Ended\n')
    