```markdown
# Binance Futures 5-Minute Strategy Backtester & Data Fetcher

> A high-performance quantitative trading framework featuring real-time Binance data retrieval and ultra-fast NVIDIA GPU-accelerated (CuPy) backtesting across various moving average, gap, RSI, and Trailing Stop-Loss (TSL) strategies.

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Accelerated by](https://img.shields.io/badge/Accelerated%20by-NVIDIA%20CUDA%20%2F%20CuPy-green.svg)](https://cupy.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Table of Contents
- [About the Repository](#-about-the-repository)
- [File Briefings & Script Breakdown](#-file-briefings--script-breakdown)
- [Prerequisites](#-prerequisites)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🚀 About the Repository

This repository combines a live Binance USD-M Futures data fetching utility with a series of high-speed GPU backtesting scripts[cite: 10, 11]. By using **CUDA Raw Kernels (`cp.RawKernel`)**, the framework parallelizes parameter optimization loops for 5-minute BTCUSDT OHLC datasets, cutting calculation times from hours (as programs run on CPU) down to seconds[cite: 11].

---

## 📂 File Briefings & Script Breakdown

Here is a summary of what each Python file in this repository accomplishes:

| Filename | Description |
| :--- | :--- |
| `5min ohlc BINANACE FUTURES data fetcher.py` | Connects to the Binance USD-M Futures API to continuously poll and pull live 5-minute OHLC candle data for `BTCUSDT` at regular intervals[cite: 10]. |
| `cuda 5min v1 gap strategy.py` | Backtests an EMA crossover strategy incorporating custom long/short EMA deviations and gap exit rules computed directly on the GPU[cite: 11]. |
| `cuda 5min v2 entry and exit variation strategy.py` | Expands optimization parameters by splitting EMA variations into distinct entry and exit threshold criteria for long and short trades[cite: 12]. |
| `cuda 5min v3 simple close exit strategy.py` | Tests a streamlined model where positions close immediately when the price crosses back over the 34 EMA baseline[cite: 13]. |
| `cuda 5min v4 simple crossover exit strategy.py` | Evaluates performance using standard 5 EMA and 34 EMA line crossover signals to dictate trade exits[cite: 14]. |
| `cuda 5min v5 no deviations.py` | Focuses strictly on entry and exit variations without secondary EMA deviation filters, simplifying the parameter grid space[cite: 15]. |
| `cuda 5min v6 with tsl.py` | Introduces a Trailing Stop-Loss (`TSL`) mechanism alongside dynamic entry/exit thresholds to lock in profits during active trades[cite: 16]. |
| `cuda 5min v7 using rsi and tsl.py` | Integrates a 14-period Relative Strength Index (`RSI`) filter into entry conditions alongside TSL and EMA indicators[cite: 17]. |
| `cuda 5min v8 upgraded version 5 without deviations.py` | Uses Simple Moving Averages (`SMA`) instead of exponential averages alongside TSL logic for comparative trend testing[cite: 18]. |
| `cuda 5min v9.py` | Refined high-speed kernel configuration optimizing SMA/EMA variation rules and automatic final-candle trade closures[cite: 19]. |
| `cuda 5min v10.py` | Refined GPU backtester utilizing a 5 EMA and 13 SMA crossover system, incorporating consecutive trend-bar checks and strict entry variation boundaries[cite: 20]. |

---

## ⚙️ Prerequisites

Ensure your system meets the following requirements before execution:
* **Operating System:** Windows 10/11 or Linux
* **Hardware:** An NVIDIA GPU supporting CUDA capability 
* **Python:** Version 3.8 or higher
* **CUDA Toolkit / Drivers:** Compatible with your installed CuPy environment

---

## 📦 Installation & Setup

1. **Clone the repository:**
```bash
git clone [https://github.com/pandiyan07/5min-candle-Binance-BTC-FUT-strategy-backtester-runs-on-NVIDIA-GPU-parallel-processing-using-CUDA.git](https://github.com/pandiyan07/5min-candle-Binance-BTC-FUT-strategy-backtester-runs-on-NVIDIA-GPU-parallel-processing-using-CUDA.git)
cd 5min-candle-Binance-BTC-FUT-strategy-backtester-runs-on-NVIDIA-GPU-parallel-processing-using-CUDA

```

2. **Create and activate a virtual environment (recommended):**
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

```


3. **Install dependencies:**
Make sure to install the CuPy package variant that matches your local CUDA version (e.g., `cupy-cuda12x`):
```bash
pip install pandas numpy pandas_ta cupy-cuda12x joblib binance-futures-connector

```



---

## 💡 Usage

1. Place your historical OHLC data file (e.g., `5min ohlc MARCH.csv` or `data_ohlc_feb1.csv`) in the root directory, or run the live fetcher script.
2. Execute any of the backtesting scripts via command line:
```bash
python "cuda 5min v6 with tsl.py"

```


3. Performance summaries will print directly to the console, and sorted CSV output files (such as `net_profit.csv`) will be automatically generated.

---

## 🗂️ Project Structure

```text
your-repo-name/
│
├── cache/                                      # Joblib calculation cache
├── 5min ohlc MARCH.csv                         # Sample 5-min OHLC dataset
├── 5min ohlc BINANACE FUTURES data fetcher.py  # Live API polling script
├── cuda 5min v1 gap strategy.py                # GPU backtester v1
├── cuda 5min v6 with tsl.py                    # GPU backtester with TSL
├── cuda 5min v7 using rsi and tsl.py           # RSI & TSL integrated backtester
├── cuda 5min v9.py                             # optimizing SMA/EMA variation
├── cuda 5min v10.py                            # Latest optimized GPU backtester
├── .gitignore
└── README.md

```

---

## 🤝 Contributing

Contributions, enhancements, and strategy adjustments are welcome! Feel free to open an issue or submit a pull request.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.