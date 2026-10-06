import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import pulp

# =========================================================
# PHASE 1: Clean Data Engineering & Calendar Derivation
# =========================================================
df = pd.read_csv("transactions_in_usd.csv")
df['Transaction Date'] = pd.to_datetime(df['Transaction Date'], format='mixed')

# Isolate Airport ATM and sort chronologically
atm_df = df[df['ATM Name'] == 'Airport ATM'].copy()
atm_df = atm_df.sort_values('Transaction Date').set_index('Transaction Date')

# Restrict to the most recent continuous block (avoiding ancient multi-week missing gaps)
atm_df = atm_df.loc['2015-01-01':'2017-12-09']
atm_df = atm_df.asfreq('D')

# Handle minor missing withdrawals via rolling median or localized interpolation instead of flat global lines
atm_df['Total amount Withdrawn'] = atm_df['Total amount Withdrawn'].interpolate(method='time')

# BUG 3 FIX: Derive calendar attributes programmatically from the timestamp
atm_df['Day_Name'] = atm_df.index.day_name()
atm_df['Is_Weekend'] = atm_df.index.weekday >= 5  # Saturday=5, Sunday=6
atm_df['Working_Day'] = np.where(atm_df['Is_Weekend'], 'H', 'W')

withdrawals = atm_df['Total amount Withdrawn']
exog_vars = pd.get_dummies(atm_df['Working_Day'], prefix='Working', drop_first=True).astype(float)

# ADF Test on weekly differenced series to verify D=1
weekly_diff = withdrawals.diff(7).dropna()
adf_weekly = adfuller(weekly_diff)
print(f"--- Stationarity Check (Weekly Differenced) ---")
print(f"ADF Statistic: {adf_weekly[0]:.4f} | p-value: {adf_weekly[1]:.4f} (Stationary)\n")

# =========================================================
# PHASE 2: Corrected SARIMAX (d=0, D=1, s=7)
# =========================================================
print("Fitting corrected SARIMAX model (d=0, D=1, s=7)...")
model_sarimax = sm.tsa.SARIMAX(
    withdrawals,
    exog=exog_vars,
    order=(1, 0, 1),            # d=0 (fixed over-differencing)
    seasonal_order=(1, 1, 1, 7), # D=1, s=7 (captures weekly cycle)
    enforce_stationarity=False,
    enforce_invertibility=False
)
results = model_sarimax.fit(disp=False)

# Create future exogenous variables for the next 7 days based on actual calendar days
days = 7
future_dates = pd.date_range(start=withdrawals.index[-1] + pd.Timedelta(days=1), periods=days)
future_is_weekend = future_dates.weekday >= 5
future_working = np.where(future_is_weekend, 'H', 'W')
future_exog = pd.get_dummies(pd.Series(future_working), prefix='Working', drop_first=True).reindex(columns=exog_vars.columns, fill_value=0.0)
future_exog.index = future_dates

# Generate forecast and extract 95% Upper Confidence Bound dynamically
forecast = results.get_forecast(steps=days, exog=future_exog)
confidence_intervals = forecast.conf_int(alpha=0.05)

forecasted_demand = confidence_intervals.iloc[:, 1].astype(float).tolist()
print(f"Corrected 7-Day Risk Bounds: {[round(x, 2) for x in forecasted_demand]}\n")

# =========================================================
# PHASE 3: PuLP Optimization (MILP with Truck Capacity)
# =========================================================
holding_cost_rate = 0.05 / 365 
truck_dispatch_cost = 150      
max_atm_capacity = 150000      
max_truck_capacity = 50000    
initial_inventory = 5000      

prob = pulp.LpProblem("Single_ATM_Cash_Optimization_Corrected", pulp.LpMinimize)

x = pulp.LpVariable.dicts("Delivery", range(days), lowBound=0, cat='Continuous')
y = pulp.LpVariable.dicts("Truck_Dispatched", range(days), cat='Binary')
I = pulp.LpVariable.dicts("Inventory", range(days), lowBound=0, cat='Continuous')

prob += pulp.lpSum([holding_cost_rate * I[t] + truck_dispatch_cost * y[t] for t in range(days)])

for t in range(days):
    if t == 0:
        prob += I[t] == initial_inventory + x[t] - forecasted_demand[t]
    else:
        prob += I[t] == I[t-1] + x[t] - forecasted_demand[t]
    
    prob += I[t] <= max_atm_capacity
    prob += x[t] <= max_truck_capacity * y[t]

prob.solve()

plot_inventory = [I[t].varValue for t in range(days)]
plot_deliveries = [x[t].varValue for t in range(days)]

print("--- Optimal Delivery Schedule (Corrected Model) ---")
for t in range(days):
    truck_sent = 'Yes' if y[t].varValue == 1 else 'No'
    print(f"Day {t+1}: Deliver ${plot_deliveries[t]:,.2f} | End Inventory: ${plot_inventory[t]:,.2f} | Truck Sent: {truck_sent}")
    
print(f"\nTotal Minimum Cost: ${pulp.value(prob.objective):,.2f}")

# =========================================================
# PHASE 4: Visualization
# =========================================================
plot_days = [f'Day {t+1}' for t in range(days)]

plt.figure(figsize=(12, 6))
plt.bar(plot_days, forecasted_demand, color='#ff9999', label='Forecasted Max Demand (Risk Bound)')
plt.plot(plot_days, plot_inventory, marker='o', color='#2ca02c', linewidth=3, markersize=8, label='End-of-Day Inventory')

for i, d in enumerate(plot_deliveries):
    if d > 0:
        plt.annotate(f'Truck Arrives:\n+${d:,.0f}', 
                     xy=(i, plot_inventory[i]), 
                     xytext=(i, plot_inventory[i] + 15000),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                     ha='center', fontsize=9, fontweight='bold')

plt.title('Corrected Optimal Cash Routing Schedule (Airport ATM)', fontsize=14, fontweight='bold')
plt.ylabel('Cash Volume ($)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()