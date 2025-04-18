#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys

def visualize_radiation_pattern(pattern_file, output_file=None):
    """Visualize a radiation pattern from an H5 file"""
    # Open the H5 file
    with h5py.File(pattern_file, 'r') as f:
        pattern = f['radiation_pattern'][:]
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    plt.contourf(pattern, cmap='viridis', levels=50)
    plt.colorbar(label='Gain (normalized)')
    plt.title('Antenna Radiation Pattern')
    plt.xlabel('Azimuth Angle (°)')
    plt.ylabel('Elevation Angle (°)')
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return pattern

def visualize_s_parameters(s_param_file, output_file=None):
    """Visualize S-parameters from a CSV file"""
    # Load the S-parameters
    data = pd.read_csv(s_param_file)
    frequencies = data['frequency']
    s11 = data['s11']
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, s11)
    plt.axhline(y=-10, color='r', linestyle='--', label='Return Loss = -10 dB')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('S11 (dB)')
    plt.title('Antenna Return Loss (S11)')
    plt.grid(True)
    plt.legend()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return frequencies, s11

def analyze_parameters(params_file, output_dir=None):
    """Analyze antenna parameters from a CSV file"""
    # Load parameters
    data = pd.read_csv(params_file)
    
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Visualize gain distribution
    plt.figure(figsize=(10, 6))
    plt.hist(data['gain'], bins=15, alpha=0.7)
    plt.axvline(x=3, color='r', linestyle='--', label='Min Target (3 dBi)')
    plt.axvline(x=5, color='g', linestyle='--', label='Max Target (5 dBi)')
    plt.xlabel('Gain (dBi)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Antenna Gain')
    plt.legend()
    plt.grid(True)
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'gain_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    # Visualize SAR distribution
    plt.figure(figsize=(10, 6))
    plt.hist(data['sar'], bins=15, alpha=0.7)
    plt.axvline(x=1.6, color='r', linestyle='--', label='Regulatory Threshold (1.6 W/kg)')
    plt.xlabel('SAR (W/kg)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Specific Absorption Rate (SAR)')
    plt.legend()
    plt.grid(True)
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'sar_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    # Visualize relationship between bending radius and gain
    plt.figure(figsize=(10, 6))
    plt.scatter(data['bending_radius'], data['gain'], alpha=0.7)
    plt.axvline(x=10, color='r', linestyle='--', label='Comfort Threshold (10 mm)')
    plt.xlabel('Bending Radius (mm)')
    plt.ylabel('Gain (dBi)')
    plt.title('Effect of Bending Radius on Antenna Gain')
    plt.legend()
    plt.grid(True)
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'bending_vs_gain.png'), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    # Visualize relationship between bending radius and SAR
    plt.figure(figsize=(10, 6))
    plt.scatter(data['bending_radius'], data['sar'], alpha=0.7)
    plt.axvline(x=10, color='r', linestyle='--', label='Comfort Threshold (10 mm)')
    plt.axhline(y=1.6, color='g', linestyle='--', label='SAR Threshold (1.6 W/kg)')
    plt.xlabel('Bending Radius (mm)')
    plt.ylabel('SAR (W/kg)')
    plt.title('Effect of Bending Radius on SAR')
    plt.legend()
    plt.grid(True)
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'bending_vs_sar.png'), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 50)
    print(f"Total antenna designs: {len(data)}")
    print(f"Gain range: {data['gain'].min():.2f} to {data['gain'].max():.2f} dBi")
    print(f"SAR range: {data['sar'].min():.2f} to {data['sar'].max():.2f} W/kg")
    print(f"Bending radius range: {data['bending_radius'].min():.2f} to {data['bending_radius'].max():.2f} mm")
    
    # Calculate compliance percentage
    gain_compliant = ((data['gain'] >= 3) & (data['gain'] <= 5)).sum()
    gain_pct = gain_compliant / len(data) * 100
    
    sar_compliant = (data['sar'] <= 1.6).sum()
    sar_pct = sar_compliant / len(data) * 100
    
    bend_compliant = (data['bending_radius'] >= 10).sum()
    bend_pct = bend_compliant / len(data) * 100
    
    print("\nCompliance with Target Metrics:")
    print("-" * 50)
    print(f"Gain (3-5 dBi): {gain_compliant} designs ({gain_pct:.1f}%)")
    print(f"SAR (≤1.6 W/kg): {sar_compliant} designs ({sar_pct:.1f}%)")
    print(f"Bending radius (≥10 mm): {bend_compliant} designs ({bend_pct:.1f}%)")
    
    # Calculate overall compliance (all criteria met)
    all_compliant = ((data['gain'] >= 3) & (data['gain'] <= 5) & 
                     (data['sar'] <= 1.6) & 
                     (data['bending_radius'] >= 10)).sum()
    all_pct = all_compliant / len(data) * 100
    
    print(f"All criteria met: {all_compliant} designs ({all_pct:.1f}%)")
    
    return data

def main():
    # Set default data paths
    data_dir = os.path.join('data', 'raw')
    params_file = os.path.join(data_dir, 'antenna_params.csv')
    plots_dir = 'plots'
    
    # Check if the paths exist
    if not os.path.exists(params_file):
        print(f"Error: Antenna parameters file not found at {params_file}")
        print("Please run the data generation script first.")
        sys.exit(1)
    
    # Create plots directory
    os.makedirs(plots_dir, exist_ok=True)
    
    # Analyze parameter distributions
    print("Analyzing antenna parameters...")
    data = analyze_parameters(params_file, plots_dir)
    
    # Visualize first sample's radiation pattern and S-parameters
    if len(data) > 0:
        first_sample = data.iloc[0]
        pattern_file = os.path.join(data_dir, 'patterns', first_sample['pattern_file'])
        s_param_file = os.path.join(data_dir, 's_parameters', first_sample['s11_file'])
        
        if os.path.exists(pattern_file):
            print("\nVisualizing radiation pattern for first sample...")
            visualize_radiation_pattern(pattern_file, os.path.join(plots_dir, 'sample_radiation_pattern.png'))
        
        if os.path.exists(s_param_file):
            print("Visualizing S-parameters for first sample...")
            visualize_s_parameters(s_param_file, os.path.join(plots_dir, 'sample_s_parameters.png'))
    
    print(f"\nVisualization complete. Plots saved to '{plots_dir}' directory.")

if __name__ == "__main__":
    main() 