import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_metrics(csv_path, output_path):
    df = pd.read_csv(csv_path)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    config_name = Path(csv_path).stem
    
    # Loss plot
    ax1.plot(df['epoch'], df['loss_train'], label='Train Loss', color='steelblue')
    if 'loss_val' in df.columns and not df['loss_val'].isnull().all():
        ax1.plot(df['epoch'], df['loss_val'], label='Val Loss', color='tomato')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Loss vs Epochs ({config_name})')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Accuracy plot
    ax2.plot(df['epoch'], df['acc_train'], label='Train Acc', color='steelblue')
    if 'acc_val' in df.columns and not df['acc_val'].isnull().all():
        ax2.plot(df['epoch'], df['acc_val'], label='Val Acc', color='tomato')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'Accuracy vs Epochs ({config_name})')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120)
    print(f"Plot saved to {output_path}")
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True, help='Path to metrics CSV')
    parser.add_argument('--out', type=str, required=True, help='Path to output PNG')
    args = parser.parse_args()
    plot_metrics(args.csv, args.out)
