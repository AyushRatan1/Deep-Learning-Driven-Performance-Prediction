import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add the models directory to path for importing enhanced calculations
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))

try:
    from enhanced_calculations import EnhancedAntennaCalculator, FrequencyBandManager, SARAnalysisEngine
except ImportError:
    print("Warning: Enhanced calculations module not found. Using fallback calculations.")
    EnhancedAntennaCalculator = None

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

def generate_antenna_params_enhanced(num_samples, output_dir, frequency_ghz=2.45):
    """
    Generate enhanced synthetic antenna parameter metadata using physics-based calculations.
    
    Parameters:
    -----------
    num_samples : int
        Number of antenna designs
    output_dir : str
        Directory to save metadata  
    frequency_ghz : float
        Operating frequency in GHz
    """
    # Try to import enhanced calculations, fallback to simplified if not available
    try:
        from ..models.enhanced_calculations import EnhancedAntennaCalculator, FrequencyBandManager, SARAnalysisEngine
        enhanced_available = True
        print(f"Using enhanced physics-based calculations for {num_samples} samples at {frequency_ghz} GHz")
    except ImportError:
        enhanced_available = False
        print(f"Enhanced calculations not available, using simplified calculations")
    
    # Initialize lists for metadata
    data = []
    
    for i in range(num_samples):
        # Generate random physical parameters first (realistic ranges for wearable antennas)
        substrate_thickness = np.random.uniform(0.5, 3.2)  # mm (0.5-3.2mm typical for PCB)
        substrate_permittivity = np.random.uniform(2.2, 10.2)  # Typical range for RF substrates
        patch_width = np.random.uniform(5, 50)  # mm (reasonable for wearable)
        patch_length = np.random.uniform(5, 50)  # mm  
        feed_position = np.random.uniform(1, min(patch_width, patch_length)/4)  # mm
        bending_radius = np.random.uniform(10, 100)  # mm (wearable comfort)
        conductor_thickness = np.random.uniform(17.5, 70)  # μm (0.5oz to 2oz copper)
        
        if enhanced_available:
            # Use enhanced physics-based calculations
            
            # Calculate realistic antenna gain using proper patch antenna theory
            efficiency = EnhancedAntennaCalculator.calculate_efficiency(
                frequency_ghz, substrate_permittivity, substrate_thickness, conductor_thickness
            )
            
            gain_dbi = EnhancedAntennaCalculator.calculate_patch_antenna_gain(
                frequency_ghz, patch_width, patch_length, substrate_permittivity, 
                substrate_thickness, efficiency
            )
            
            # Calculate bandwidth using Q-factor method
            bandwidth_mhz = EnhancedAntennaCalculator.calculate_bandwidth(
                frequency_ghz, patch_width, substrate_permittivity, substrate_thickness
            )
            
            # Calculate SAR using proper physics: SAR = σ|E|²/ρ
            # Assume 100mW power and 10mm distance for wearable scenario
            power_w = 0.1
            distance_m = 0.01
            gain_linear = 10**(gain_dbi / 10)
            power_density_calc = power_w * gain_linear / (4 * np.pi * distance_m**2)
            e_field_rms = np.sqrt(power_density_calc * EnhancedAntennaCalculator.ETA_0)
            
            # Calculate SAR for multiple tissue types and depths
            sar_skin_surface = EnhancedAntennaCalculator.calculate_sar_physics_based(
                e_field_rms, frequency_ghz, 'skin', 0
            )
            sar_skin_1mm = EnhancedAntennaCalculator.calculate_sar_physics_based(
                e_field_rms, frequency_ghz, 'skin', 1
            )
            sar_fat_surface = EnhancedAntennaCalculator.calculate_sar_physics_based(
                e_field_rms, frequency_ghz, 'fat', 0
            )
            sar_muscle_surface = EnhancedAntennaCalculator.calculate_sar_physics_based(
                e_field_rms, frequency_ghz, 'muscle', 0
            )
            
            # Safety assessments against multiple standards
            fcc_safety = EnhancedAntennaCalculator.assess_sar_safety(sar_skin_surface, 'fcc')
            icnirp_safety = EnhancedAntennaCalculator.assess_sar_safety(sar_skin_surface, 'icnirp')
            
            # Calculate 3D radiation pattern characteristics
            radiation_pattern_max = gain_dbi
            radiation_pattern_3db_beamwidth = 80 + np.random.normal(0, 10)  # Typical beamwidth
            
            # Calculate resonant frequency based on physical dimensions
            c = EnhancedAntennaCalculator.C
            effective_length = patch_length / 1000  # Convert to meters
            effective_permittivity = (substrate_permittivity + 1)/2 + (substrate_permittivity - 1)/2 * \
                                   (1 + 12 * (substrate_thickness/1000) / (patch_width/1000))**(-0.5)
            
            resonant_freq_calculated = c / (2 * effective_length * np.sqrt(effective_permittivity)) / 1e9
            
            # Add some realistic variation and noise
            gain_dbi += np.random.normal(0, 0.2)
            sar_skin_surface += np.random.normal(0, sar_skin_surface * 0.1)  # 10% variation
            bandwidth_mhz += np.random.normal(0, bandwidth_mhz * 0.05)  # 5% variation
            
        else:
            # Fallback to enhanced simplified calculations
            wavelength_mm = 300 / frequency_ghz  # Approximate wavelength in mm
            
            # More realistic gain calculation based on antenna theory
            optimal_length = wavelength_mm / (2 * np.sqrt(substrate_permittivity))
            size_factor = np.exp(-((patch_length - optimal_length)**2) / (optimal_length**2))
            substrate_factor = 1.0 + 0.1 * np.log(substrate_permittivity)
            thickness_factor = 1.0 + 0.2 * (substrate_thickness / wavelength_mm)
            
            gain_dbi = 2 + 8 * size_factor * substrate_factor * thickness_factor
            gain_dbi += np.random.normal(0, 0.5)
            gain_dbi = np.clip(gain_dbi, -2, 12)
            
            # More realistic SAR calculation with frequency dependence
            freq_factor = (frequency_ghz / 2.45)**0.7  # Frequency scaling
            substrate_loss = substrate_permittivity * np.sqrt(frequency_ghz) * 0.02
            proximity_factor = np.exp(-bending_radius / 20)
            
            sar_base = 0.8 + 0.6 * np.random.random()
            sar_skin_surface = sar_base * freq_factor * (1 + substrate_loss) * (1 + proximity_factor)
            sar_skin_surface = np.clip(sar_skin_surface, 0.1, 3.0)
            
            # Estimate other SAR values
            sar_skin_1mm = sar_skin_surface * 0.8  # Attenuated at depth
            sar_fat_surface = sar_skin_surface * 0.3  # Lower conductivity
            sar_muscle_surface = sar_skin_surface * 1.2  # Higher conductivity
            
            # Bandwidth calculation
            Q_factor = 50 + 20 * substrate_permittivity / substrate_thickness
            bandwidth_mhz = (frequency_ghz * 1000) / Q_factor
            
            # Efficiency estimation
            efficiency = 0.6 + 0.3 * np.exp(-substrate_thickness / wavelength_mm) + np.random.normal(0, 0.05)
            efficiency = np.clip(efficiency, 0.3, 0.95)
            
            # Radiation pattern characteristics
            radiation_pattern_max = gain_dbi
            radiation_pattern_3db_beamwidth = 70 + np.random.normal(0, 15)
            
            # Safety assessments
            fcc_safety = {'compliant': sar_skin_surface < 1.6, 'safety_margin_percent': (1.6 - sar_skin_surface) / 1.6 * 100}
            icnirp_safety = {'compliant': sar_skin_surface < 2.0, 'safety_margin_percent': (2.0 - sar_skin_surface) / 2.0 * 100}
            
            # Resonant frequency estimation
            resonant_freq_calculated = frequency_ghz * (1 + np.random.normal(0, 0.05))
        
        # Common parameters for both enhanced and simplified calculations
        vswr = 1.2 + np.random.exponential(0.3)  # Realistic VSWR distribution
        vswr = np.clip(vswr, 1.1, 3.0)
        
        return_loss_db = -20 * np.log10((vswr + 1) / (vswr - 1))
        
        # Compile all parameters
        antenna_data = {
            'id': f'antenna_{i:04d}',
            'frequency_ghz': frequency_ghz,
            'substrate_thickness': substrate_thickness,
            'substrate_permittivity': substrate_permittivity,
            'patch_width': patch_width,
            'patch_length': patch_length,
            'feed_position': feed_position,
            'bending_radius': bending_radius,
            'conductor_thickness': conductor_thickness,
            'gain_dbi': gain_dbi,
            'efficiency': efficiency,
            'bandwidth_mhz': bandwidth_mhz,
            'sar_skin_surface': sar_skin_surface,
            'sar_skin_1mm': sar_skin_1mm,
            'sar_fat_surface': sar_fat_surface,
            'sar_muscle_surface': sar_muscle_surface,
            'vswr': vswr,
            'return_loss_db': return_loss_db,
            'radiation_pattern_max': radiation_pattern_max,
            'radiation_pattern_3db_beamwidth': radiation_pattern_3db_beamwidth,
            'resonant_frequency': resonant_freq_calculated,
            'fcc_compliant': fcc_safety['compliant'],
            'icnirp_compliant': icnirp_safety['compliant'],
            'fcc_safety_margin': fcc_safety['safety_margin_percent'],
            'icnirp_safety_margin': icnirp_safety['safety_margin_percent'],
            'pattern_file': f'pattern_{i:04d}.csv',
            's11_file': f's11_{i:04d}.csv'
        }
        
        data.append(antenna_data)
        
        # Generate corresponding radiation pattern and S11 data
        if enhanced_available:
            # Generate frequency sweep data using enhanced calculations
            try:
                freq_sweep_data = SARAnalysisEngine.generate_sar_vs_frequency_data(
                    patch_width, patch_length, substrate_thickness, substrate_permittivity,
                    power_w=0.1, min_freq_ghz=frequency_ghz*0.8, max_freq_ghz=frequency_ghz*1.2, num_points=50
                )
                
                # Save frequency sweep data
                freq_sweep_filename = os.path.join(output_dir, f'freq_sweep_{i:04d}.csv')
                freq_sweep_data.to_csv(freq_sweep_filename, index=False)
                antenna_data['freq_sweep_file'] = f'freq_sweep_{i:04d}.csv'
                
                # Generate circular SAR map
                sar_map_data = SARAnalysisEngine.generate_circular_sar_map(
                    frequency_ghz, patch_width, patch_length, power_w=0.1, map_size_mm=100, resolution=25
                )
                
                # Save SAR map data as JSON
                import json
                sar_map_filename = os.path.join(output_dir, f'sar_map_{i:04d}.json')
                with open(sar_map_filename, 'w') as f:
                    json.dump(sar_map_data, f, indent=2)
                antenna_data['sar_map_file'] = f'sar_map_{i:04d}.json'
                
            except Exception as e:
                print(f"Warning: Could not generate advanced data for antenna {i}: {e}")
        
        # Generate simplified radiation pattern data
        generate_simplified_pattern_data(i, output_dir, gain_dbi, radiation_pattern_3db_beamwidth)
        
        # Generate simplified S11 data
        generate_simplified_s11_data(i, output_dir, frequency_ghz, bandwidth_mhz, return_loss_db)
        
        if (i + 1) % 50 == 0:
            print(f"Generated {i + 1}/{num_samples} antenna designs...")
    
    # Save enhanced antenna parameters
    df = pd.DataFrame(data)
    params_file = os.path.join(output_dir, 'antenna_params_enhanced.csv')
    df.to_csv(params_file, index=False)
    
    print(f"\nEnhanced antenna parameter generation completed!")
    print(f"Generated {num_samples} antenna designs at {frequency_ghz} GHz")
    print(f"Data saved to: {params_file}")
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"Gain (dBi): {df['gain_dbi'].mean():.2f} ± {df['gain_dbi'].std():.2f}")
    print(f"SAR Skin (W/kg): {df['sar_skin_surface'].mean():.3f} ± {df['sar_skin_surface'].std():.3f}")
    print(f"Efficiency: {df['efficiency'].mean():.2f} ± {df['efficiency'].std():.2f}")
    print(f"Bandwidth (MHz): {df['bandwidth_mhz'].mean():.1f} ± {df['bandwidth_mhz'].std():.1f}")
    print(f"FCC Compliant: {df['fcc_compliant'].sum()}/{len(df)} ({df['fcc_compliant'].mean()*100:.1f}%)")
    print(f"ICNIRP Compliant: {df['icnirp_compliant'].sum()}/{len(df)} ({df['icnirp_compliant'].mean()*100:.1f}%)")
    
    return params_file


def generate_simplified_pattern_data(antenna_id, output_dir, gain_dbi, beamwidth_deg):
    """Generate simplified radiation pattern data"""
    theta = np.linspace(0, 180, 181)  # Elevation angles
    phi = np.linspace(0, 360, 361)    # Azimuth angles
    
    # Simplified pattern based on gain and beamwidth
    pattern_data = []
    for t in theta:
        for p in phi:
            # Simple cosine pattern with beamwidth control
            pattern_factor = np.cos(np.radians(t)) if t <= 90 else 0
            beamwidth_factor = np.exp(-((t - 90)**2) / (2 * (beamwidth_deg/2.35)**2))  # Gaussian
            
            gain_linear = 10**(gain_dbi/10) * pattern_factor * beamwidth_factor
            gain_db = 10 * np.log10(max(gain_linear, 1e-6))
            
            pattern_data.append({
                'theta': t,
                'phi': p,
                'gain_db': gain_db,
                'gain_linear': gain_linear
            })
    
    # Save pattern data
    pattern_df = pd.DataFrame(pattern_data)
    pattern_filename = os.path.join(output_dir, f'pattern_{antenna_id:04d}.csv')
    pattern_df.to_csv(pattern_filename, index=False)


def generate_simplified_s11_data(antenna_id, output_dir, center_freq_ghz, bandwidth_mhz, return_loss_db):
    """Generate simplified S11 parameter data"""
    # Generate frequency sweep around center frequency
    freq_span = bandwidth_mhz * 3 / 1000  # 3x bandwidth in GHz
    frequencies = np.linspace(center_freq_ghz - freq_span/2, center_freq_ghz + freq_span/2, 200)
    
    s11_data = []
    for freq in frequencies:
        # Simple resonator model
        freq_offset = (freq - center_freq_ghz) / (bandwidth_mhz / 1000)
        s11_magnitude = return_loss_db / (1 + freq_offset**2)  # Lorentzian shape
        s11_magnitude = max(s11_magnitude, -50)  # Limit minimum S11
        
        s11_data.append({
            'frequency_ghz': freq,
            'frequency_hz': freq * 1e9,
            's11_db': s11_magnitude,
            's11_magnitude': 10**(s11_magnitude/20)
        })
    
    # Save S11 data
    s11_df = pd.DataFrame(s11_data)
    s11_filename = os.path.join(output_dir, f's11_{antenna_id:04d}.csv')
    s11_df.to_csv(s11_filename, index=False)


def generate_frequency_sweep_data(output_dir, antenna_params=None):
    """
    Generate enhanced SAR vs frequency data for extended range up to 150 GHz
    """
    if not EnhancedAntennaCalculator:
        print("Enhanced calculations not available. Skipping frequency sweep.")
        return
    
    if antenna_params is None:
        antenna_params = {
            'patch_width_mm': 10.0,
            'patch_length_mm': 12.0,
            'substrate_thickness_mm': 1.6,
            'substrate_er': 4.4,
            'power_w': 0.1
        }
    
    print("Generating comprehensive frequency sweep data (0.5 - 150 GHz)...")
    
    # Generate comprehensive SAR vs frequency data using enhanced calculations
    sar_data = SARAnalysisEngine.generate_sar_vs_frequency_data(
        **antenna_params,
        min_freq_ghz=0.5,
        max_freq_ghz=150.0,
        num_points=500
    )
    
    # Add safety compliance analysis across frequencies
    sar_data['fcc_compliant'] = sar_data['sar_skin_surface'] <= 1.6
    sar_data['icnirp_compliant'] = sar_data['sar_skin_surface'] <= 2.0
    sar_data['safety_status'] = sar_data.apply(lambda row: 
        'safe' if row['sar_skin_surface'] <= 0.8 else
        'caution' if row['sar_skin_surface'] <= 1.6 else
        'unsafe', axis=1)
    
    # Calculate safety margins
    sar_data['fcc_safety_margin_percent'] = ((1.6 - sar_data['sar_skin_surface']) / 1.6) * 100
    sar_data['icnirp_safety_margin_percent'] = ((2.0 - sar_data['sar_skin_surface']) / 2.0) * 100
    
    # Identify frequency bands of interest
    sar_data['frequency_band'] = sar_data['frequency_ghz'].apply(lambda f:
        'UHF' if f < 3 else
        'S-band' if f < 4 else
        'C-band' if f < 8 else
        'X-band' if f < 12 else
        'Ku-band' if f < 18 else
        'K-band' if f < 27 else
        'Ka-band' if f < 40 else
        'mmWave' if f < 95 else
        'THz'
    )
    
    # Save the enhanced data
    freq_sweep_file = os.path.join(output_dir, 'enhanced_sar_frequency_sweep.csv')
    sar_data.to_csv(freq_sweep_file, index=False)
    
    print(f"Enhanced frequency sweep data saved to {freq_sweep_file}")
    print(f"Frequency range: {sar_data['frequency_ghz'].min():.1f} - {sar_data['frequency_ghz'].max():.1f} GHz")
    print(f"SAR range: {sar_data['sar_skin_surface'].min():.4f} - {sar_data['sar_skin_surface'].max():.4f} W/kg")
    print(f"FCC compliance: {sar_data['fcc_compliant'].sum()}/{len(sar_data)} frequencies ({sar_data['fcc_compliant'].mean()*100:.1f}%)")
    
    # Generate summary statistics by frequency band
    band_stats = sar_data.groupby('frequency_band').agg({
        'frequency_ghz': ['min', 'max'],
        'sar_skin_surface': ['min', 'max', 'mean'],
        'fcc_compliant': 'mean',
        'icnirp_compliant': 'mean'
    }).round(4)
    
    print("\nSAR Analysis by Frequency Band:")
    print(band_stats)
    
    return sar_data


def generate_circular_sar_maps(output_dir, num_maps=10):
    """
    Generate circular SAR maps for different antenna configurations and frequencies
    """
    if not EnhancedAntennaCalculator:
        print("Enhanced calculations not available. Skipping SAR maps.")
        return
    
    print(f"Generating {num_maps} circular SAR maps...")
    
    # Test frequencies covering different bands
    test_frequencies = [2.45, 5.8, 10, 24, 38, 60, 95, 120, 140, 150]
    
    maps_data = []
    
    for i in range(num_maps):
        # Select frequency and antenna parameters
        frequency_ghz = test_frequencies[i % len(test_frequencies)]
        
        # Random antenna configuration
        patch_width = np.random.uniform(8, 15)
        patch_length = np.random.uniform(8, 15) 
        power_w = np.random.uniform(0.05, 0.2)  # 50-200mW
        
        print(f"Generating map {i+1}/{num_maps} for {frequency_ghz} GHz...")
        
        # Generate circular SAR map using enhanced physics
        map_data = SARAnalysisEngine.generate_circular_sar_map(
            frequency_ghz, patch_width, patch_length, power_w,
            map_size_mm=100, resolution=50
        )
        
        # Extract key metrics
        max_sar = map_data['maxSAR']
        avg_sar = map_data['avgSAR']
        
        # Safety assessment
        fcc_compliant = max_sar <= 1.6
        icnirp_compliant = max_sar <= 2.0
        
        # Store metadata
        map_metadata = {
            'map_id': f'circular_map_{i:04d}',
            'frequency_ghz': frequency_ghz,
            'patch_width_mm': patch_width,
            'patch_length_mm': patch_length,
            'power_w': power_w,
            'map_size_mm': map_data['mapSize'],
            'resolution': map_data['resolution'],
            'max_sar': max_sar,
            'avg_sar': avg_sar,
            'fcc_compliant': fcc_compliant,
            'icnirp_compliant': icnirp_compliant,
            'fcc_safety_margin_percent': ((1.6 - max_sar) / 1.6) * 100,
            'icnirp_safety_margin_percent': ((2.0 - max_sar) / 2.0) * 100,
            'safe_area_percentage': np.sum(np.array(map_data['data']) <= 0.8) / (map_data['resolution']**2) * 100,
            'data_file': f'sar_maps/circular_map_{i:04d}.csv'
        }
        
        maps_data.append(map_metadata)
        
        # Save individual map data
        os.makedirs(os.path.join(output_dir, 'sar_maps'), exist_ok=True)
        map_file = os.path.join(output_dir, 'sar_maps', f'circular_map_{i:04d}.csv')
        
        # Convert 2D array to DataFrame for saving
        map_df = pd.DataFrame(map_data['data'])
        map_df.to_csv(map_file, index=False, header=False)
    
    # Save maps metadata
    maps_df = pd.DataFrame(maps_data)
    metadata_file = os.path.join(output_dir, 'circular_sar_maps_metadata.csv')
    maps_df.to_csv(metadata_file, index=False)
    
    print(f"Circular SAR maps metadata saved to {metadata_file}")
    print(f"Individual map data saved in {os.path.join(output_dir, 'sar_maps')}")
    print(f"\nSAR Maps Summary:")
    print(f"Max SAR range: {maps_df['max_sar'].min():.4f} - {maps_df['max_sar'].max():.4f} W/kg")
    print(f"FCC compliance: {maps_df['fcc_compliant'].sum()}/{len(maps_df)} maps ({maps_df['fcc_compliant'].mean()*100:.1f}%)")
    print(f"Average safe area: {maps_df['safe_area_percentage'].mean():.1f}%")
    
    return maps_df

def generate_circular_sar_map_data(frequency=2.45, power_mw=100, tissue_type='skin', map_size_km=1.0, resolution=100):
    """
    Generate circular SAR map data for spatial analysis around antenna
    
    Parameters:
    -----------
    frequency : float
        Frequency in GHz
    power_mw : float
        Power in milliwatts
    tissue_type : str
        Type of tissue ('skin', 'fat', 'muscle')
    map_size_km : float
        Map size in kilometers
    resolution : int
        Grid resolution
    
    Returns:
    --------
    dict : SAR map data with spatial coordinates in km
    """
    # Enhanced tissue properties based on frequency
    tissue_props = {
        'skin': {
            2.45: {'sigma': 1.46, 'density': 1100},
            5.8: {'sigma': 3.717, 'density': 1100},
            10: {'sigma': 6.2, 'density': 1100},
            24: {'sigma': 12.1, 'density': 1100},
            60: {'sigma': 22.8, 'density': 1100},
            150: {'sigma': 48.5, 'density': 1100}
        },
        'fat': {
            2.45: {'sigma': 0.101, 'density': 920},
            5.8: {'sigma': 0.29, 'density': 920},
            10: {'sigma': 0.45, 'density': 920},
            24: {'sigma': 1.05, 'density': 920},
            60: {'sigma': 2.8, 'density': 920},
            150: {'sigma': 6.2, 'density': 920}
        },
        'muscle': {
            2.45: {'sigma': 1.74, 'density': 1040},
            5.8: {'sigma': 4.96, 'density': 1040},
            10: {'sigma': 8.2, 'density': 1040},
            24: {'sigma': 16.8, 'density': 1040},
            60: {'sigma': 32.8, 'density': 1040},
            150: {'sigma': 65.8, 'density': 1040}
        }
    }
    
    # Get tissue properties for given frequency
    tissue_data = tissue_props.get(tissue_type, tissue_props['skin'])
    frequencies = list(tissue_data.keys())
    closest_freq = min(frequencies, key=lambda x: abs(x - frequency))
    props = tissue_data[closest_freq]
    
    # Generate spatial grid in km
    x = np.linspace(-map_size_km/2, map_size_km/2, resolution)
    y = np.linspace(-map_size_km/2, map_size_km/2, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Calculate distance from antenna (at center)
    distance_km = np.sqrt(X**2 + Y**2)
    distance_m = distance_km * 1000  # Convert to meters
    
    # Antenna parameters (simplified patch antenna)
    gain_dbi = 6 + np.random.normal(0, 1)  # 6 dBi nominal gain
    gain_linear = 10**(gain_dbi / 10)
    power_w = power_mw / 1000  # Convert to watts
    
    # Calculate power density using far-field approximation
    # For near-field (distance < λ), apply correction factor
    wavelength_m = 0.3 / frequency
    sar_map = np.zeros_like(distance_m)
    
    for i in range(resolution):
        for j in range(resolution):
            dist = distance_m[i, j]
            if dist < 0.001:  # Very close to antenna
                sar_map[i, j] = 0
                continue
                
            # Power density calculation
            power_density = (power_w * gain_linear) / (4 * np.pi * dist**2)
            
            # Near-field enhancement for distances < 2λ
            if dist < 2 * wavelength_m:
                power_density *= (2 * wavelength_m / dist)**0.5
            
            # Electric field
            e_field = np.sqrt(power_density * 377)  # Free space impedance
            
            # SAR calculation: SAR = σ|E|²/ρ
            sar = props['sigma'] * e_field**2 / props['density']
            
            # Apply distance-based attenuation for biological tissue
            attenuation_factor = np.exp(-0.1 * dist * frequency)  # Simplified attenuation
            sar *= attenuation_factor
            
            sar_map[i, j] = max(0, min(sar, 10))  # Clamp to reasonable values
    
    # Calculate statistics
    max_sar = np.max(sar_map)
    avg_sar = np.mean(sar_map[sar_map > 0])
    
    # Safety assessment
    fcc_limit = 1.6  # W/kg
    icnirp_limit = 2.0  # W/kg
    
    fcc_compliant = max_sar <= fcc_limit
    icnirp_compliant = max_sar <= icnirp_limit
    
    # Determine overall safety status
    if max_sar <= fcc_limit:
        safety_status = 'safe'
    elif max_sar <= icnirp_limit:
        safety_status = 'caution'
    else:
        safety_status = 'unsafe'
    
    # Calculate spatial statistics
    safe_area_km2 = np.sum(sar_map <= fcc_limit) * (map_size_km / resolution)**2
    total_area_km2 = map_size_km**2
    safe_percentage = (safe_area_km2 / total_area_km2) * 100
    
    return {
        'sar_map': sar_map.tolist(),
        'x_coords': x.tolist(),
        'y_coords': y.tolist(),
        'frequency': frequency,
        'power_mw': power_mw,
        'tissue_type': tissue_type,
        'map_size_km': map_size_km,
        'resolution': resolution,
        'statistics': {
            'max_sar': float(max_sar),
            'avg_sar': float(avg_sar),
            'safe_area_km2': float(safe_area_km2),
            'total_area_km2': float(total_area_km2),
            'safe_percentage': float(safe_percentage)
        },
        'safety_assessment': {
            'fcc_compliant': fcc_compliant,
            'icnirp_compliant': icnirp_compliant,
            'safety_status': safety_status,
            'fcc_limit': fcc_limit,
            'icnirp_limit': icnirp_limit,
            'recommendations': [
                f'Maximum SAR: {max_sar:.4f} W/kg',
                f'Safe area coverage: {safe_percentage:.1f}%',
                'FCC compliant' if fcc_compliant else 'Exceeds FCC limit',
                'ICNIRP compliant' if icnirp_compliant else 'Exceeds ICNIRP limit'
            ]
        }
    }

def main():
    parser = argparse.ArgumentParser(description='Generate enhanced synthetic data for antenna modeling')
    parser.add_argument('--num_samples', type=int, default=500, help='Number of samples to generate')
    parser.add_argument('--output_dir', type=str, default='data/raw', help='Output directory')
    parser.add_argument('--frequency', type=float, default=2.45, help='Primary frequency in GHz')
    parser.add_argument('--enhanced', action='store_true', help='Use enhanced physics-based calculations')
    parser.add_argument('--generate_sweep', action='store_true', help='Generate frequency sweep data')
    parser.add_argument('--generate_maps', action='store_true', help='Generate SAR maps')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    patterns_dir = os.path.join(args.output_dir, 'patterns')
    s_params_dir = os.path.join(args.output_dir, 's_parameters')
    os.makedirs(patterns_dir, exist_ok=True)
    os.makedirs(s_params_dir, exist_ok=True)
    
    # Generate data
    print(f"====== Enhanced Antenna Performance Prediction Data Generation ======")
    print(f"Starting enhanced workflow execution at {pd.Timestamp.now()}")
    
    if args.enhanced or EnhancedAntennaCalculator:
        print("\nGenerating enhanced physics-based antenna parameters...")
        df = generate_antenna_params_enhanced(args.num_samples, args.output_dir, args.frequency)
        
        if args.generate_sweep:
            print("\nGenerating frequency sweep analysis...")
            generate_frequency_sweep_data(args.output_dir)
        
        if args.generate_maps:
            print("\nGenerating circular SAR maps...")
            generate_circular_sar_maps(args.output_dir)
    else:
        print("\nGenerating standard synthetic data...")
        generate_radiation_patterns(args.num_samples, patterns_dir, 128)
        generate_s_parameters(args.num_samples, s_params_dir, freq_points=101)
        df = generate_antenna_params(args.num_samples, args.output_dir)
    
    # Generate radiation patterns and S-parameters for enhanced data too
    if args.enhanced or EnhancedAntennaCalculator:
        print("\nGenerating radiation patterns...")
        generate_radiation_patterns(args.num_samples, patterns_dir, 128)
        
        print("Generating S-parameters...")
        generate_s_parameters(args.num_samples, s_params_dir, freq_points=101)
    
    print(f"\n====== Data generation completed at {pd.Timestamp.now()} ======")
    print(f"Enhanced data saved to: {args.output_dir}")
    
    if 'df' in locals():
        print(f"\nGenerated {len(df)} antenna designs")
        print("Summary statistics:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        print(df[numeric_cols].describe())

if __name__ == "__main__":
    main() 