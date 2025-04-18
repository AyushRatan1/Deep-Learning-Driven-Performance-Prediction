import os
import numpy as np
import pandas as pd
import h5py
import argparse
from pathlib import Path

def generate_radiation_patterns(num_samples, output_dir, resolution=128):
    """
    Generate synthetic radiation patterns for antenna designs.
    
    Parameters:
    -----------
    num_samples : int
        Number of patterns to generate
    output_dir : str
        Directory to save generated patterns
    resolution : int
        Resolution of pattern images
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(num_samples):
        # Generate a random radiation pattern
        # Main lobe with random direction and beamwidth
        theta_center = np.random.uniform(0, 2*np.pi)
        phi_center = np.random.uniform(0, np.pi)
        beamwidth = np.random.uniform(0.2, 1.0)
        
        # Create theta and phi grids
        theta = np.linspace(0, 2*np.pi, resolution)
        phi = np.linspace(0, np.pi, resolution)
        THETA, PHI = np.meshgrid(theta, phi)
        
        # Calculate angular distance from center
        angular_dist = np.sqrt(
            (np.sin(PHI) * np.cos(THETA) - np.sin(phi_center) * np.cos(theta_center))**2 +
            (np.sin(PHI) * np.sin(THETA) - np.sin(phi_center) * np.sin(theta_center))**2 +
            (np.cos(PHI) - np.cos(phi_center))**2
        )
        
        # Create Gaussian-shaped main lobe
        gain = np.exp(-angular_dist**2 / (2 * beamwidth**2))
        
        # Add some random side lobes and nulls
        num_sidelobes = np.random.randint(3, 10)
        for _ in range(num_sidelobes):
            sl_theta = np.random.uniform(0, 2*np.pi)
            sl_phi = np.random.uniform(0, np.pi)
            sl_width = np.random.uniform(0.05, 0.2)
            sl_gain = np.random.uniform(0.1, 0.4)
            
            sl_dist = np.sqrt(
                (np.sin(PHI) * np.cos(THETA) - np.sin(sl_phi) * np.cos(sl_theta))**2 +
                (np.sin(PHI) * np.sin(THETA) - np.sin(sl_phi) * np.sin(sl_theta))**2 +
                (np.cos(PHI) - np.cos(sl_phi))**2
            )
            
            gain += sl_gain * np.exp(-sl_dist**2 / (2 * sl_width**2))
        
        # Normalize
        gain = gain / np.max(gain)
        
        # Add some noise
        noise = np.random.normal(0, 0.02, gain.shape)
        gain = gain + noise
        gain = np.clip(gain, 0, 1)
        
        # Save in HDF5 format
        file_path = os.path.join(output_dir, f"pattern_{i:04d}.h5")
        with h5py.File(file_path, 'w') as f:
            f.create_dataset('radiation_pattern', data=gain)
            
        # Also save a CSV version for flexibility
        csv_path = os.path.join(output_dir, f"pattern_{i:04d}.csv")
        pd.DataFrame(gain).to_csv(csv_path, index=False)

def generate_s_parameters(num_samples, output_dir, freq_points=101, ism_band='2.4GHz'):
    """
    Generate synthetic S-parameters for antenna designs.
    
    Parameters:
    -----------
    num_samples : int
        Number of S-parameter sets to generate
    output_dir : str
        Directory to save generated S-parameters
    freq_points : int
        Number of frequency points
    ism_band : str
        ISM band - either '2.4GHz' or '5.8GHz'
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define frequency range based on ISM band
    if ism_band == '2.4GHz':
        freq_start = 2.2  # GHz
        freq_end = 2.6  # GHz
        center_freq = 2.45  # Center frequency
    elif ism_band == '5.8GHz':
        freq_start = 5.5  # GHz
        freq_end = 6.1  # GHz
        center_freq = 5.8  # Center frequency
    else:
        raise ValueError(f"Unknown ISM band: {ism_band}")
    
    # Generate frequency points
    frequencies = np.linspace(freq_start, freq_end, freq_points)
    
    for i in range(num_samples):
        # Generate random resonance parameters
        resonance_freq = np.random.normal(center_freq, 0.05)
        q_factor = np.random.uniform(10, 50)
        depth = np.random.uniform(-30, -15)  # dB
        
        # Generate S11 curve
        s11 = depth * np.exp(-((frequencies - resonance_freq)**2) / (2 * (resonance_freq / q_factor)**2))
        
        # Add some random ripples
        num_ripples = np.random.randint(2, 6)
        for _ in range(num_ripples):
            ripple_freq = np.random.uniform(freq_start, freq_end)
            ripple_width = np.random.uniform(0.01, 0.05)
            ripple_depth = np.random.uniform(-5, -1)
            
            s11 += ripple_depth * np.exp(-((frequencies - ripple_freq)**2) / (2 * ripple_width**2))
        
        # Offset to a realistic baseline
        baseline = np.random.uniform(-8, -5)
        s11 += baseline
        
        # Add some noise
        noise = np.random.normal(0, 0.5, s11.shape)
        s11 += noise
        
        # Limit extremes
        s11 = np.clip(s11, -35, 0)
        
        # Save as CSV
        data = pd.DataFrame({
            'frequency': frequencies,
            's11': s11
        })
        file_path = os.path.join(output_dir, f"s11_{i:04d}.csv")
        data.to_csv(file_path, index=False)

def generate_antenna_params(num_samples, output_dir, ism_band='2.4GHz'):
    """
    Generate synthetic antenna parameter metadata.
    
    Parameters:
    -----------
    num_samples : int
        Number of antenna designs
    output_dir : str
        Directory to save metadata
    ism_band : str
        ISM band - either '2.4GHz' or '5.8GHz'
    """
    # Initialize lists for metadata
    data = []
    
    for i in range(num_samples):
        # Generate random physical parameters first
        substrate_thickness = np.random.uniform(0.5, 3.0)  # mm
        substrate_permittivity = np.random.uniform(1.5, 4.5)
        patch_width = np.random.uniform(30, 50)  # mm
        patch_length = np.random.uniform(30, 50)  # mm
        feed_position = np.random.uniform(5, 15)  # mm
        bending_radius = np.random.uniform(5, 30)  # mm
        
        # Generate performance parameters based on physical parameters
        # These relationships are simplified but try to maintain physical realism
        
        # Calculate wavelength based on ISM band
        if ism_band == '2.4GHz':
            wavelength = 125  # mm (approx for 2.4 GHz)
        else:
            wavelength = 52  # mm (approx for 5.8 GHz)
        
        # Gain is influenced by patch size, substrate, and bending
        optimal_length = wavelength / np.sqrt(substrate_permittivity) / 2
        size_factor = np.exp(-(patch_length - optimal_length)**2 / 100)
        bending_factor = 1 - np.exp(-bending_radius / 10)
        
        gain = 3 + 2 * size_factor * bending_factor + np.random.normal(0, 0.3)
        gain = np.clip(gain, 1, 6)  # Realistic range for textile antennas
        
        # SAR is influenced by substrate thickness and permittivity
        sar_base = 1.2 + 0.5 * np.random.random()
        thickness_factor = np.exp(-substrate_thickness / 2)
        
        sar = sar_base * thickness_factor + np.random.normal(0, 0.1)
        sar = np.clip(sar, 0.5, 2.5)  # Typical range for wearable antennas
        
        # Efficiency is based on similar factors
        efficiency = 60 + 20 * size_factor * (1 - thickness_factor) + np.random.normal(0, 5)
        efficiency = np.clip(efficiency, 40, 90)  # Realistic range for textile antennas
        
        # Add entry to data
        entry = {
            'id': i,
            'pattern_file': f"pattern_{i:04d}.h5",
            's11_file': f"s11_{i:04d}.csv",
            'substrate_thickness': substrate_thickness,
            'substrate_permittivity': substrate_permittivity,
            'patch_width': patch_width,
            'patch_length': patch_length,
            'feed_position': feed_position,
            'bending_radius': bending_radius,
            'gain': gain,
            'sar': sar,
            'efficiency': efficiency,
            'ism_band': ism_band
        }
        
        data.append(entry)
    
    # Create DataFrame and save as CSV
    df = pd.DataFrame(data)
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "antenna_params.csv"), index=False)
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic data for antenna modeling')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to generate')
    parser.add_argument('--output_dir', type=str, default='../data/raw', help='Output directory')
    parser.add_argument('--ism_band', type=str, default='2.4GHz', choices=['2.4GHz', '5.8GHz'], 
                        help='ISM band to simulate')
    parser.add_argument('--resolution', type=int, default=128, help='Resolution of radiation patterns')
    
    args = parser.parse_args()
    
    # Create output directories
    patterns_dir = os.path.join(args.output_dir, 'patterns')
    s_params_dir = os.path.join(args.output_dir, 's_parameters')
    
    # Generate data
    print(f"Generating {args.num_samples} synthetic antenna designs...")
    
    print("Generating radiation patterns...")
    generate_radiation_patterns(args.num_samples, patterns_dir, args.resolution)
    
    print("Generating S-parameters...")
    generate_s_parameters(args.num_samples, s_params_dir, ism_band=args.ism_band)
    
    print("Generating antenna parameters metadata...")
    df = generate_antenna_params(args.num_samples, args.output_dir, ism_band=args.ism_band)
    
    print(f"Data generation complete! Files saved to {args.output_dir}")
    print(f"Generated {len(df)} antenna designs with the following parameter ranges:")
    
    for col in df.columns:
        if col not in ['id', 'pattern_file', 's11_file', 'ism_band']:
            print(f"{col}: {df[col].min():.2f} to {df[col].max():.2f}")

if __name__ == '__main__':
    main() 