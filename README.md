# ATM Cash Demand Forecasting & Replenishment Optimisation

An end-to-end forecasting-to-optimisation pipeline for ATM cash logistics. A seasonal time-series model forecasts the next 7 days of cash withdrawals at an ATM, and a mixed-integer linear program (MILP) uses those forecasts to decide **when to send a cash truck and how much to deliver**, at minimum total holding and dispatch cost.

---

## Overview

Cash logistics is a trade-off: stocking too much cash is costly to hold, while stocking too little risks the ATM running dry. This project connects the two halves of the decision:

1. **Predict** – a SARIMAX model forecasts daily withdrawals and produces a 95% upper bound for each day, so the plan is built against a high-demand scenario.
2. **Optimise** – a MILP chooses the delivery schedule that covers that demand at the lowest cost, subject to ATM and truck capacity.

---

## Dataset

The project uses `transactions_in_usd.csv`, a daily record of withdrawals for five ATMs (Big Street, Mount Road, Airport, KK Nagar and Christ College).

| Column | Description |
|---|---|
| `ATM Name` | ATM identifier |
| `Transaction Date` | Date of the record |
| `No Of Withdrawals` | Number of withdrawals |
| `No Of XYZ Card Withdrawals`, `No Of Other Card Withdrawals` | Withdrawals split by card type |
| `Total amount Withdrawn` | Total cash withdrawn (USD) |
| `Amount withdrawn XYZ Card`, `Amount withdrawn Other Card` | Amount split by card type |
| `Weekday` | Day of the week |
| `Festival Religion` | Festival indicator |
| `Working Day` | Working day / holiday flag |
| `Holiday Sequence` | Position in a holiday sequence |

This project models the **Airport ATM** using `Total amount Withdrawn`.

---

## Project structure

```
.
├── Proj.py                    # Full pipeline: data prep, forecasting, optimisation, plot
├── transactions_in_usd.csv    # ATM transaction data
└── README.md
```

---

## Pipeline

### Phase 1 – Data engineering
- Filters the data to the Airport ATM and sorts it chronologically.
- Uses the continuous block from 2015-01-01 to 2017-12-09 and reindexes it to a daily frequency.
- Fills any missing days in `Total amount Withdrawn` with time-based interpolation.
- Derives calendar features from the timestamp: day name, weekend flag, and a working-day (`W`) / holiday (`H`) indicator used as an exogenous regressor.
- Runs an **Augmented Dickey-Fuller test** on the weekly-differenced series to confirm stationarity after seasonal differencing (`D = 1`).

### Phase 2 – SARIMAX forecasting
- **Model:** `SARIMAX(1, 0, 1)(1, 1, 1, 7)` with the working-day indicator as an exogenous variable.
  - Non-seasonal part: `d = 0`
  - Seasonal part: `D = 1`, period `s = 7` to capture the weekly cycle
- Builds future exogenous values for the next 7 days from the actual calendar.
- Forecasts 7 days ahead and takes the **upper bound of the 95% confidence interval** as the planning demand for each day.

### Phase 3 – MILP optimisation (PuLP)
Minimises total cost over the 7-day horizon.

**Decision variables**
- `x[t]` – cash delivered on day `t` (continuous, ≥ 0)
- `y[t]` – whether a truck is dispatched on day `t` (binary)
- `I[t]` – end-of-day inventory on day `t` (continuous, ≥ 0)

**Objective**

```
minimise  Σ_t ( holding_cost_rate · I[t]  +  truck_dispatch_cost · y[t] )
```

**Constraints**
- Inventory balance: `I[t] = I[t-1] + x[t] − forecasted_demand[t]` (starting from the initial inventory)
- ATM capacity: `I[t] ≤ max_atm_capacity`
- Truck capacity and linking: `x[t] ≤ max_truck_capacity · y[t]`

**Parameters**

| Parameter | Value |
|---|---|
| Holding cost rate | 5% per year (applied daily) |
| Truck dispatch cost | $150 per trip |
| ATM capacity | $150,000 |
| Truck capacity | $50,000 |
| Initial inventory | $5,000 |

### Phase 4 – Visualisation
Plots the forecasted peak demand as bars and end-of-day inventory as a line, with arrows marking the days a truck arrives and the amount delivered.

---

## Getting started

### 1. Install dependencies

```bash
pip install pandas numpy matplotlib statsmodels pulp
```

### 2. Run

```bash
python Proj.py
```

Make sure `transactions_in_usd.csv` is in the same folder as `Proj.py`.

---

## Output

The script prints:
- The ADF test statistic and p-value for the weekly-differenced series
- The 7-day demand risk bounds
- The optimal delivery schedule: delivery amount, end-of-day inventory and whether a truck is sent, for each day
- The total minimum cost

It then displays the cash routing chart for the 7-day plan.

---

## Tech stack

- **Python**
- **pandas / NumPy** – data handling and calendar features
- **statsmodels** – ADF test and SARIMAX forecasting
- **PuLP** – mixed-integer linear programming
- **Matplotlib** – visualisation

---

## Key concepts

- **Seasonal time-series modelling** with weekly seasonality and exogenous calendar regressors
- **Stationarity testing** with the Augmented Dickey-Fuller test
- **Prediction-interval-based planning**, using the 95% upper bound as a risk-aware demand
- **Mixed-integer linear programming** with fixed-charge (truck dispatch) costs and capacity constraints
- **Predict-then-optimise** workflow linking forecasting directly to an operational decision
