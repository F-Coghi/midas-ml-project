# MIDAS Mixed-Frequency Models (NN + Nonlinear Baselines)

This project addresses a mixed-frequency econometric problem: how to use
higher-frequency data to nowcast and forecast lower-frequency
macroeconomic indicators such as unemployment and vacancies. Rather than
aggregating monthly signals into quarterly averages, we employ a MIDAS
framework that directly incorporates higher-frequency lags into a
low-frequency regression (see Valkanov (2006a); Ghysels and Valkanov (2006b); 
Ghysels, Sinko and Valkanov(2007); Ghysels et al. (2007); 
Clements and Galvão (2008)). 

To avoid restrictive parametric lag assumptions, we estimate lag weights 
using a small feedforward neural network, preserving linearity in 
observables while allowing temporal information decay patterns.
We compare this structured semiparametric approach with fully nonlinear 
sequence models (MLP, RNN, LSTM, RandomForest) to assess whether economic 
discipline or black-box flexibility yields better predictive
performance.

References: 
- Ghysels, E., Santa-Clara, P., & Valkanov, R. (2006). 
Predicting volatility: Getting the most out of return data sampled at 
different frequencies. Journal of Econometrics, 131, 59–95.

- Ghysels, G., & Valkanov, R. (2006). Linear time series processes 
with mixed data sampling and MIDAS regression models [Mimeo].

- Ghysels, G., Sinko, A., & Valkanov, R. (2007). MIDAS regressions: 
Further results and new directions. Econometric Reviews, 26, 
53–90.

- Clements, M. P., & Galvão, A. B. (2008). Macroeconomic forecasting 
with mixed-frequency data: Forecasting output growth in the United States. 
Journal of Business and Economic Statistics, 26 (4), 546–554.



This repository contains two scripts for forecasting a quarterly target
series using a monthly regressor in a **true mixed-frequency MIDAS
design**:

-   `midas_nn.py`: linear MIDAS regression where lag weights are learned
    by small neural nets (softmax-normalised, nonnegative, sum-to-1).
-   `nonlinear-baselines.py`: nonlinear baselines (MLP / RNN / LSTM /
    RandomForest) trained on the same data.

Both scripts: - read the same input format (tab-separated text), - build
the mixed-frequency lag design (quarterly lags of Y + monthly lags of
X), - use a time split (default 90/10) and standardise using **training
statistics only**, - report metrics and plot predictions vs actual.

------------------------------------------------------------------------

## Repository structure

Recommended layout:

. ├── midas_nn.py ├── nonlinear-baselines.py ├── data/ │ └──
test_StockemployeesvsGVA.txt ├── requirements.txt └── README.md

------------------------------------------------------------------------

## Setup

Create a virtual environment and install dependencies:

``` bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

------------------------------------------------------------------------

## Data format

Input is a **2-column tab-separated file** with a **header**:

-   Column 0: `X` (monthly, always present)
-   Column 1: `Y` (quarterly, present only at quarter months; blank
    otherwise)

Example (schematic):

    X   Y
    1.23    10.0
    1.10
    0.98
    1.05    10.4
    ...

------------------------------------------------------------------------

## Usage

### 1) MIDAS-NN (learned lag weights)

``` bash
python midas_nn.py --data data/test_StockemployeesvsGVA.txt --N 30 --M 80
```

Useful options (see `--help` in the script): - `--train-frac` (default:
0.9) - `--epochs`, `--lr`, `--hidden`, `--restarts`, `--seed` -
`--device` (if present in your script; otherwise it auto-detects)

------------------------------------------------------------------------

### 2) Nonlinear baselines (MLP / RNN / LSTM / RandomForest)

``` bash
python nonlinear-baselines.py --data data/test_StockemployeesvsGVA.txt --N 4 --M 12
```

Defaults to a **90/10** time split unless you override:

``` bash
python nonlinear-baselines.py --data data/test_StockemployeesvsGVA.txt --N 4 --M 12 --train-frac 0.9
```

------------------------------------------------------------------------

## Notes

-   Both scripts assume a **time-ordered** dataset: do not shuffle rows.
-   The train/test split is chronological.
-   The test set includes the last training sample (`split-1:`) to keep
    plots visually continuous at the boundary.

------------------------------------------------------------------------

## License

MIT

------------------------------------------------------------------------

## Citation

If you use or adapt this code, please cite the repository.
···
@article{ghysels2006predicting,
  title={Predicting volatility: getting the most out of return data sampled at different frequencies},
  author={Ghysels, Eric and Santa-Clara, Pedro and Valkanov, Rossen},
  journal={Journal of Econometrics},
  volume={131},
  number={1-2},
  pages={59--95},
  year={2006},
  publisher={Elsevier}
}
···
