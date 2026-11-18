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

def CALCULATE_SMA(df, length):
    df['oc2'] = (df['open'].astype(float) + df['close'].astype(float)) / 2
    return ta.sma(df['oc2'], length=length)

# GPU Kernel for Backtesting
BACKTEST_KERNEL = cp.RawKernel(
    r'''
    extern "C" __global__
    void BACKTEST_KERNEL(
        const float* close_price,
        const float* ema_5, const float* sma_34,
        const float* params,
        float* results, 
        const int num_candles,
        const int num_params)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= num_params) return;

        float sma34_LONG_ENTRY_variation = params[tid * 4];
        float sma34_SHORT_ENTRY_variation = params[tid * 4 + 1];
        float sma34_LONG_EXIT_variation = params[tid * 4 + 2];
        float sma34_SHORT_EXIT_variation = params[tid * 4 + 3];

        int result_idx = tid * 15;
        float long_total_profit = 0.0;
        float long_total_loss = 0.0;
        float short_total_profit = 0.0;
        float short_total_loss = 0.0;
        int num_long_trades = 0;
        int num_short_trades = 0;
        float position = 0.0;
        float entry_price = 0.0;

        for (int j = 2; j < num_candles; j++) {
            float close = close_price[j];

            bool long_condition = (ema_5[j] > sma_34[j]) && ((sma_34[j] - sma_34[j-5]) > sma34_LONG_ENTRY_variation);
            bool short_condition = (ema_5[j] < sma_34[j]) && ((sma_34[j-5] - sma_34[j]) > sma34_SHORT_ENTRY_variation);

            if (position == 0.0) {
                if (long_condition) {
                    position = 1.0;
                    entry_price = close;
                    num_long_trades++;
                } else if (short_condition) {
                    position = 2.0;
                    entry_price = close;
                    num_short_trades++;
                }
            } else if (position == 1.0) {
                if ((sma_34[j-5] - sma_34[j]) > sma34_LONG_EXIT_variation) {
                    float profit = close - entry_price;
                    (profit > 0) ? long_total_profit += profit : long_total_loss += profit; 
                    position = 0.0;
                }
            } else if (position == 2.0) {
                if ((sma_34[j] - sma_34[j-5]) > sma34_SHORT_EXIT_variation) {
                    float profit = entry_price - close;
                    (profit > 0) ? short_total_profit += profit : short_total_loss += profit;
                    position = 0.0;
                }
            }
            if (j == num_candles - 1 && position != 0.0) {  
                float profit = (position == 1.0) ? (close - entry_price) : (entry_price - close);
                (profit > 0) ? long_total_profit += profit : long_total_loss += profit;
                position = 0.0;  // Reset position after closing
            }
        }

        results[result_idx] = sma34_LONG_ENTRY_variation;
        results[result_idx + 1] = sma34_SHORT_ENTRY_variation; 
        results[result_idx + 2] = sma34_LONG_EXIT_variation;
        results[result_idx + 3] = sma34_SHORT_EXIT_variation;
        results[result_idx + 4] = num_long_trades + num_short_trades;
        results[result_idx + 5] = num_long_trades;
        results[result_idx + 6] = num_short_trades;
        results[result_idx + 7] = (num_long_trades + num_short_trades > 0) ? ((long_total_profit + short_total_profit) / ((long_total_profit + short_total_profit) + fabsf(long_total_loss + short_total_loss))) * 100.0f : 0.0f;
        results[result_idx + 8] = long_total_profit;
        results[result_idx + 9] = long_total_loss;
        if ((long_total_profit) > fabsf(long_total_loss)) {
            results[result_idx + 10] = (fabsf(long_total_loss) / (long_total_profit)) * 100.0f;
        } else {
            results[result_idx + 10] = 0.0f;
        }
        results[result_idx + 11] = short_total_profit;
        results[result_idx + 12] = short_total_loss;
        if ((short_total_profit) > fabsf(short_total_loss)) {
            results[result_idx + 13] = (fabsf(short_total_loss) / (short_total_profit)) * 100.0f;
        } else {
            results[result_idx + 13] = 0.0f;
        }
        results[result_idx + 14] = (long_total_profit + short_total_profit) - fabsf(long_total_loss + short_total_loss); 
    }
        ''',
    'BACKTEST_KERNEL'
)

def BACKTEST_GPU(df, params):
    num_candles = len(df)
    num_params = len(params)
    results = cp.zeros((num_params * 15), dtype=cp.float32)
    threads_per_block = 256
    blocks_per_grid = (num_params + threads_per_block - 1) // threads_per_block

    gpu_data = {
        'close': cp.asarray(df['close'], dtype=cp.float32),
        'ema_5': cp.asarray(df['ema_5'], dtype=cp.float32),
        'sma_34': cp.asarray(df['sma_34'], dtype=cp.float32),
        'params': cp.asarray(params, dtype=cp.float32)
    }
    
    BACKTEST_KERNEL((blocks_per_grid,), (threads_per_block,), (
        gpu_data['close'],
        gpu_data['ema_5'],
        gpu_data['sma_34'],
        gpu_data['params'],
        results,
        num_candles,
        num_params
    ))
    return results.get().reshape((num_params, 15))

def TRADING_BOT(df):
    df['ema_5'] = CALCULATE_EMA(df, 5)
    df['sma_34'] = CALCULATE_SMA(df, 34)
    
    params = np.array([
        (sma34_LONG_ENTRY_variation, sma34_SHORT_ENTRY_variation, sma34_LONG_EXIT_variation, sma34_SHORT_EXIT_variation)
        for sma34_LONG_ENTRY_variation in range(100, 800, 100)
        for sma34_SHORT_ENTRY_variation in range(100, 800, 100)
        for sma34_LONG_EXIT_variation in range(0, 350, 50)
        for sma34_SHORT_EXIT_variation in range(0, 350, 50)
    ], dtype=np.float32)
    
    start_time = datetime.now()
    results = BACKTEST_GPU(df, params)
    end_time = datetime.now()
    
    print(f'Backtesting completed in {end_time - start_time} seconds')
    df_results = pd.DataFrame(results, columns=[
        'SMA34 Long ENTRY variation', 
        'SMA34 Short ENTRY variation',
        'SMA34 Long EXIT variation', 
        'SMA34 Short EXIT variation',
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
    OUTPUT_FILE_NET_PROFIT = r"5min ohlc MARCH v9.csv"
    
    df = pd.read_csv(INPUT_FILE)
    output_df = TRADING_BOT(df)
    
    # Sort by net profit (descending) to see the best-performing parameter combinations.
    output_df_net_profit = output_df.sort_values('Net Profit', ascending=False)
    output_df_net_profit.to_csv(OUTPUT_FILE_NET_PROFIT, index=False)
    
    table_generation_end_time = datetime.now()
    print('\nTop Highest net profit =')
    print(output_df_net_profit)
    print('\nTrading Bot Ended\n')