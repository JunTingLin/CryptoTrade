#!/usr/bin/env python3
"""
Trading Results Visualization Tool

This script parses run_agent.py log files and creates trend plots for:
- Trading actions over time
- Daily returns over time  
- Cumulative returns over time
- Portfolio net worth over time

Usage in Python/Jupyter:
    from plot_trading_results import plot_trading_data
    plot_trading_data('logs/your_log_file.out')

Usage in command line:
    python plot_trading_results.py logs/your_log_file.out
"""

import re
import matplotlib.pyplot as plt
import pandas as pd

def parse_log_file(log_file_path):
    """Parse the log file and extract trading data."""
    data = {
        'dates': [],
        'actions': [],
        'today_roi': [],
        'cumulative_roi': [],
        'net_worth': [],
        'cash': [],
        'eth_held': [],
        'open_price': [],
        'step': []
    }
    
    with open(log_file_path, 'r') as f:
        content = f.read()
    
    # Pattern to match the sections
    step_pattern = r'\*{9} START STEP (\d+) \*{9}'
    action_pattern = r'\*{3} START ACTUAL ACTION \*{3}\n([-\d\.]+)\n\*{3} END ACTUAL ACTION \*{3}'
    state_pattern = r"\*{3} START STATE \*{3}\n(\{.*?\})\n\*{3} END STATE \*{3}"
    
    steps = re.findall(step_pattern, content)
    actions = re.findall(action_pattern, content)
    state_matches = re.findall(state_pattern, content, re.DOTALL)
    
    # Process actions and their corresponding states
    for i, (step, action) in enumerate(zip(steps, actions)):
        try:
            # Parse action
            action_val = float(action)
            data['actions'].append(action_val)
            data['step'].append(int(step))
            
            # Find the corresponding state (states are indexed differently)
            # State index is step + 1 because there's an initial state at index 0
            state_idx = int(step) + 1
            if state_idx < len(state_matches):
                state_str = state_matches[state_idx]
                
                # Parse state dictionary (safely)
                # Remove numpy types from the string for safe evaluation
                clean_state = state_str.replace('np.float64(', '').replace('np.int64(', '').replace(')', '')
                
                # Extract key values using regex instead of eval for safety
                # Updated regex to handle scientific notation (e.g., 1.23e-05)
                cash_match = re.search(r"'cash': ([\d\.]+)", clean_state)
                eth_held_match = re.search(r"'eth_held': ([\d\.]+)", clean_state)
                open_match = re.search(r"'open': ([\d\.]+)", clean_state)
                net_worth_match = re.search(r"'net_worth': ([\d\.]+)", clean_state)
                roi_match = re.search(r"'roi': ([-\d\.e+-]+)", clean_state)
                today_roi_match = re.search(r"'today_roi': ([-\d\.e+-]+)", clean_state)
                date_match = re.search(r"'date': '([^']+)'", clean_state)
                
                if all([cash_match, eth_held_match, open_match, net_worth_match, roi_match, today_roi_match, date_match]):
                    data['cash'].append(float(cash_match.group(1)))
                    data['eth_held'].append(float(eth_held_match.group(1)))
                    data['open_price'].append(float(open_match.group(1)))
                    data['net_worth'].append(float(net_worth_match.group(1)))
                    data['cumulative_roi'].append(float(roi_match.group(1)))
                    data['today_roi'].append(float(today_roi_match.group(1)))
                    data['dates'].append(date_match.group(1))
                else:
                    print(f"Warning: Could not parse state for step {step}")
                    # Remove the action we just added since we can't parse the state
                    data['actions'].pop()
                    data['step'].pop()
            else:
                print(f"Warning: No state found for step {step}")
                # Remove the action we just added since we can't find the state
                data['actions'].pop()
                data['step'].pop()
                
        except Exception as e:
            print(f"Error parsing step {step}: {e}")
            continue
    
    return pd.DataFrame(data)

def create_plots(df, output_prefix='trading_plot'):
    """Create visualization plots from the parsed data."""
    if df.empty:
        print("No data to plot!")
        return

    # Convert dates to datetime
    df['datetime'] = pd.to_datetime(df['dates'])

    # 1. Trading Actions Over Time
    plt.figure(figsize=(10, 6))
    plt.plot(df['datetime'], df['actions'], 'b-', marker='o', markersize=6, linewidth=2)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    plt.title('Trading Actions Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Action (-1=Sell, 1=Buy)')
    plt.grid(True, alpha=0.3)
    plt.ylim(-1.1, 1.1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot1_path = f'{output_prefix}_actions.png'
    plt.savefig(plot1_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot1_path}")
    plt.close()

    # 2. Daily Returns
    plt.figure(figsize=(10, 6))
    plt.plot(df['datetime'], df['today_roi'] * 100, 'g-', marker='o', markersize=6, linewidth=2)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    plt.title('Daily Returns', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Daily Return (%)')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot2_path = f'{output_prefix}_daily_returns.png'
    plt.savefig(plot2_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot2_path}")
    plt.close()

    # 3. Cumulative Returns
    plt.figure(figsize=(10, 6))
    plt.plot(df['datetime'], df['cumulative_roi'] * 100, 'r-', marker='o', markersize=6, linewidth=2)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    plt.title('Cumulative Returns', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return (%)')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot3_path = f'{output_prefix}_cumulative_returns.png'
    plt.savefig(plot3_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot3_path}")
    plt.close()

    # 4. Portfolio Net Worth
    plt.figure(figsize=(10, 6))
    plt.plot(df['datetime'], df['net_worth'], 'm-', marker='o', markersize=6, linewidth=2)
    plt.axhline(y=1000000, color='gray', linestyle='--', alpha=0.7, label='Initial Value')
    plt.title('Portfolio Net Worth', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Net Worth ($)')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plot4_path = f'{output_prefix}_net_worth.png'
    plt.savefig(plot4_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot4_path}")
    plt.close()
    
    # Print summary statistics
    print("\n=== Trading Summary ===")
    print(f"Total Steps: {len(df)}")
    print(f"Date Range: {df['datetime'].min().strftime('%Y-%m-%d')} to {df['datetime'].max().strftime('%Y-%m-%d')}")
    print(f"Final Return: {df['cumulative_roi'].iloc[-1]*100:.2f}%")
    print(f"Final Net Worth: ${df['net_worth'].iloc[-1]:,.2f}")
    print(f"Best Daily Return: {df['today_roi'].max()*100:.2f}%")
    print(f"Worst Daily Return: {df['today_roi'].min()*100:.2f}%")
    print(f"Average Daily Return: {df['today_roi'].mean()*100:.2f}%")
    print(f"Daily Return Std: {df['today_roi'].std()*100:.2f}%")
    
    # Calculate Sharpe ratio (assuming risk-free rate = 0)
    if df['today_roi'].std() > 0:
        sharpe_ratio = df['today_roi'].mean() / df['today_roi'].std()
        print(f"Sharpe Ratio: {sharpe_ratio:.4f}")

def plot_trading_data(log_file_path, output_prefix=None):
    """
    Main function to parse and plot trading data.

    Args:
        log_file_path (str): Path to the log file
        output_prefix (str): Prefix for output plot files (default: derived from log file name)

    Usage in Jupyter/Python:
        plot_trading_data('logs/sample_run.out')
    """
    try:
        print(f"Parsing log file: {log_file_path}")
        df = parse_log_file(log_file_path)

        if df.empty:
            print("No trading data found in the log file!")
            return

        print(f"Found {len(df)} trading steps")

        # Generate output prefix from log file name if not provided
        if output_prefix is None:
            import os
            base_name = os.path.splitext(os.path.basename(log_file_path))[0]
            output_prefix = f"plots/{base_name}"

            # Create plots directory if it doesn't exist
            os.makedirs('plots', exist_ok=True)

        create_plots(df, output_prefix)

    except FileNotFoundError:
        print(f"Error: Log file '{log_file_path}' not found!")
    except Exception as e:
        print(f"Error: {e}")

def main():
    """Command line interface"""
    import argparse

    parser = argparse.ArgumentParser(description='Visualize trading results from log files')
    parser.add_argument('log_file', help='Path to the log file')
    args = parser.parse_args()

    plot_trading_data(args.log_file)

if __name__ == "__main__":
    main()