import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    os.makedirs("results", exist_ok=True)
    filepath = "version_5_extended_backup.xlsx"
    print(f"Loading dataset: {filepath}...")
    df = pd.read_excel(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').set_index('Date')
    
    print("\n1. Calculating QA-Filtered Grid System Loss (excluding negative metering inversions)...")
    raw_loss = df['Max. Demand at eve. peak (Generation end)'] - df['Max. Demand at eve. peak (Sub-station end)']
    n_neg = int((raw_loss < 0).sum())
    df['System_Loss_MW'] = raw_loss.where(raw_loss >= 0, np.nan)
    avg_loss_mw = df['System_Loss_MW'].mean()
    avg_loss_pct = (df['System_Loss_MW'] / df['Max. Demand at eve. peak (Generation end)']).mean() * 100
    print(f"QA Peak Transmission Loss: {avg_loss_mw:.2f} MW ({avg_loss_pct:.2f}% of generation; {n_neg} negative records excluded)")
    
    plt.figure(figsize=(12, 5))
    plt.plot(df['System_Loss_MW'], color='red', alpha=0.7, label='System Loss (MW)')
    plt.axhline(avg_loss_mw, color='black', linestyle='--', label=f'Avg Loss ({avg_loss_mw:.1f} MW)')
    plt.title("Bangladesh Grid Peak System Loss (Transmission & Distribution Waste: 2019-2026)")
    plt.ylabel("Loss (MW)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/grid_system_loss.png")
    plt.close()
    
    print("\n2. Analyzing Division-wise Surplus vs. Deficit (Supply vs. Demand)...")
    divisions = ['Dhaka', 'Chattogram', 'Rajshahi', 'Khulna', 'Mymensingh', 'Rangpur', 'Barishal', 'Sylhet']
    balances = {}
    for div in divisions:
        # Supply - Demand (Positive is surplus/exporter, Negative is deficit/consumer)
        df[f'{div}_balance'] = df[f'{div}_supply'] - df[f'{div}_demand']
        balances[div] = df[f'{div}_balance'].mean()
        
    balance_df = pd.Series(balances).sort_values()
    print("Average Supply Balance by Division (MW):")
    print(balance_df)
    
    plt.figure(figsize=(10, 5))
    colors = ['red' if x < 0 else 'green' for x in balance_df.values]
    balance_df.plot(kind='bar', color=colors)
    plt.axhline(0, color='black', linestyle='-')
    plt.title("Bangladesh Division-wise Average Net Power Balance (Supply minus Demand)")
    plt.ylabel("Net Balance (MW)")
    plt.xlabel("Division")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig("results/regional_supply_deficit.png")
    plt.close()
    
    print("\n3. Analyzing Load Factor Ratio (Day-Peak vs. Evening-Peak Generation)...")
    df['Load_Ratio'] = df['Day-peak Generation (Generation end)'] / df['Evening-peak Generation (Generation end)']
    # Compute CDD
    df['CDD'] = df['Dhaka_temp'].apply(lambda t: max(0.0, t - 24.0))
    correlation = df['Load_Ratio'].corr(df['CDD'])
    print(f"Correlation between Heat Index (CDD) and Day/Eve Load Ratio: {correlation:.3f}")
    
    plt.figure(figsize=(10, 5))
    plt.scatter(df['CDD'], df['Load_Ratio'], alpha=0.4, color='orange')
    plt.title("Impact of Summer Heat (CDD) on Day/Evening Peak Generation Ratio")
    plt.xlabel("Cooling Degree Days (CDD)")
    plt.ylabel("Day-Peak / Evening-Peak Load Ratio")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/heat_load_ratio.png")
    plt.close()

    print("\nAll grid reliability analytics completed successfully! Plots saved in 'results/'.")

if __name__ == "__main__":
    main()
