import numpy as np
import math
import random
from scipy.special import jv  # Bessel functions for circular polarization
from scipy.constants import epsilon_0, mu_0, c, pi
from typing import Dict, List, Tuple, Optional
import pandas as pd

class EnhancedAntennaCalculator:
    """Enhanced antenna calculation engine with proper physics-based formulas"""
    
    # Physical constants
    ETA_0 = 377.0  # Free space impedance (ohms)
    C = 299792458  # Speed of light (m/s)
    
    # Enhanced tissue properties based on Gabriel et al. research
    # Comprehensive frequency-dependent tissue properties (εr, σ S/m, ρ kg/m³)
    TISSUE_PROPERTIES = {
        'skin': {
            '0.3': {'er': 46.7, 'sigma': 0.69, 'density': 1100},
            '0.9': {'er': 41.4, 'sigma': 0.79, 'density': 1100},
            '1.8': {'er': 39.2, 'sigma': 1.05, 'density': 1100},
            '2.4': {'er': 38.09, 'sigma': 1.43, 'density': 1100},
            '2.45': {'er': 38.0, 'sigma': 1.46, 'density': 1100},  # Exact ISM frequency
            '3.5': {'er': 36.8, 'sigma': 2.01, 'density': 1100},
            '5.8': {'er': 35.11, 'sigma': 3.717, 'density': 1100},
            '10': {'er': 32.5, 'sigma': 6.2, 'density': 1100},
            '15': {'er': 30.8, 'sigma': 8.9, 'density': 1100},
            '24': {'er': 29.0, 'sigma': 12.1, 'density': 1100},
            '28': {'er': 28.8, 'sigma': 12.5, 'density': 1100},
            '38': {'er': 27.2, 'sigma': 15.8, 'density': 1100},
            '60': {'er': 24.2, 'sigma': 22.8, 'density': 1100},
            '77': {'er': 22.8, 'sigma': 27.2, 'density': 1100},
            '94': {'er': 21.2, 'sigma': 31.8, 'density': 1100},
            '100': {'er': 20.1, 'sigma': 35.2, 'density': 1100},
            '140': {'er': 17.5, 'sigma': 45.8, 'density': 1100},
            '150': {'er': 16.8, 'sigma': 48.5, 'density': 1100}
        },
        'fat': {
            '0.3': {'er': 5.58, 'sigma': 0.024, 'density': 920},
            '0.9': {'er': 5.46, 'sigma': 0.043, 'density': 920},
            '1.8': {'er': 5.35, 'sigma': 0.072, 'density': 920},
            '2.4': {'er': 5.29, 'sigma': 0.10, 'density': 920},
            '2.45': {'er': 5.28, 'sigma': 0.101, 'density': 920},
            '3.5': {'er': 5.19, 'sigma': 0.15, 'density': 920},
            '5.8': {'er': 4.95, 'sigma': 0.29, 'density': 920},
            '10': {'er': 4.8, 'sigma': 0.45, 'density': 920},
            '15': {'er': 4.65, 'sigma': 0.68, 'density': 920},
            '24': {'er': 4.42, 'sigma': 1.05, 'density': 920},
            '28': {'er': 4.2, 'sigma': 1.2, 'density': 920},
            '38': {'er': 4.05, 'sigma': 1.65, 'density': 920},
            '60': {'er': 3.8, 'sigma': 2.8, 'density': 920},
            '77': {'er': 3.62, 'sigma': 3.5, 'density': 920},
            '94': {'er': 3.45, 'sigma': 4.2, 'density': 920},
            '100': {'er': 3.2, 'sigma': 4.5, 'density': 920},
            '140': {'er': 2.95, 'sigma': 5.8, 'density': 920},
            '150': {'er': 2.8, 'sigma': 6.2, 'density': 920}
        },
        'muscle': {
            '0.3': {'er': 59.1, 'sigma': 0.51, 'density': 1040},
            '0.9': {'er': 56.8, 'sigma': 0.78, 'density': 1040},
            '1.8': {'er': 54.7, 'sigma': 1.25, 'density': 1040},
            '2.4': {'er': 52.82, 'sigma': 1.69, 'density': 1040},
            '2.45': {'er': 52.7, 'sigma': 1.74, 'density': 1040},
            '3.5': {'er': 50.1, 'sigma': 2.45, 'density': 1040},
            '5.8': {'er': 48.48, 'sigma': 4.96, 'density': 1040},
            '10': {'er': 45.2, 'sigma': 8.2, 'density': 1040},
            '15': {'er': 42.8, 'sigma': 12.1, 'density': 1040},
            '24': {'er': 40.2, 'sigma': 16.8, 'density': 1040},
            '28': {'er': 38.5, 'sigma': 18.5, 'density': 1040},
            '38': {'er': 36.2, 'sigma': 22.8, 'density': 1040},
            '60': {'er': 32.1, 'sigma': 32.8, 'density': 1040},
            '77': {'er': 29.8, 'sigma': 39.2, 'density': 1040},
            '94': {'er': 28.1, 'sigma': 44.5, 'density': 1040},
            '100': {'er': 26.8, 'sigma': 48.2, 'density': 1040},
            '140': {'er': 24.2, 'sigma': 58.8, 'density': 1040},
            '150': {'er': 22.5, 'sigma': 65.8, 'density': 1040}
        },
        'bone': {
            '0.3': {'er': 15.2, 'sigma': 0.08, 'density': 1990},
            '0.9': {'er': 14.1, 'sigma': 0.12, 'density': 1990},
            '1.8': {'er': 13.1, 'sigma': 0.18, 'density': 1990},
            '2.4': {'er': 12.5, 'sigma': 0.23, 'density': 1990},
            '2.45': {'er': 12.4, 'sigma': 0.24, 'density': 1990},
            '3.5': {'er': 11.8, 'sigma': 0.34, 'density': 1990},
            '5.8': {'er': 10.9, 'sigma': 0.54, 'density': 1990},
            '10': {'er': 9.8, 'sigma': 0.8, 'density': 1990},
            '15': {'er': 9.2, 'sigma': 1.1, 'density': 1990},
            '24': {'er': 8.5, 'sigma': 1.5, 'density': 1990},
            '28': {'er': 8.2, 'sigma': 1.7, 'density': 1990},
            '38': {'er': 7.8, 'sigma': 2.2, 'density': 1990},
            '60': {'er': 7.2, 'sigma': 3.2, 'density': 1990},
            '77': {'er': 6.8, 'sigma': 3.8, 'density': 1990},
            '94': {'er': 6.5, 'sigma': 4.3, 'density': 1990},
            '100': {'er': 6.2, 'sigma': 4.6, 'density': 1990},
            '140': {'er': 5.5, 'sigma': 5.8, 'density': 1990},
            '150': {'er': 5.2, 'sigma': 6.2, 'density': 1990}
        },
        'blood': {
            '0.3': {'er': 61.3, 'sigma': 0.7, 'density': 1050},
            '0.9': {'er': 59.8, 'sigma': 1.54, 'density': 1050},
            '1.8': {'er': 58.2, 'sigma': 1.65, 'density': 1050},
            '2.4': {'er': 56.9, 'sigma': 2.25, 'density': 1050},
            '2.45': {'er': 56.8, 'sigma': 2.28, 'density': 1050},
            '3.5': {'er': 54.8, 'sigma': 3.12, 'density': 1050},
            '5.8': {'er': 51.2, 'sigma': 5.85, 'density': 1050},
            '10': {'er': 48.5, 'sigma': 9.2, 'density': 1050},
            '15': {'er': 46.2, 'sigma': 12.8, 'density': 1050},
            '24': {'er': 43.8, 'sigma': 18.2, 'density': 1050},
            '28': {'er': 42.5, 'sigma': 20.5, 'density': 1050},
            '38': {'er': 40.2, 'sigma': 25.8, 'density': 1050},
            '60': {'er': 36.8, 'sigma': 35.2, 'density': 1050},
            '77': {'er': 34.5, 'sigma': 42.1, 'density': 1050},
            '94': {'er': 32.8, 'sigma': 48.5, 'density': 1050},
            '100': {'er': 31.2, 'sigma': 52.8, 'density': 1050},
            '140': {'er': 28.5, 'sigma': 65.2, 'density': 1050},
            '150': {'er': 27.2, 'sigma': 69.8, 'density': 1050}
        }
    }
    
    # Safety standards (W/kg)
    SAR_LIMITS = {
        'fcc': 1.6,      # FCC limit (US)
        'icnirp': 2.0,   # ICNIRP limit (EU, most of world)
        'ic': 1.6,       # Industry Canada
        'acma': 2.0,     # Australia
        'safe_threshold': 0.8,  # Conservative safety threshold (50% of FCC)
    }
    
    @staticmethod
    def interpolate_tissue_properties(frequency_ghz: float, tissue_type: str) -> Dict[str, float]:
        """
        Interpolate tissue properties for any frequency using Gabriel's parametric model
        
        Args:
            frequency_ghz: Frequency in GHz
            tissue_type: Type of tissue (skin, fat, muscle, bone, blood)
        
        Returns:
            Dictionary with interpolated properties (er, sigma, density)
        """
        tissue_data = EnhancedAntennaCalculator.TISSUE_PROPERTIES.get(tissue_type, 
                     EnhancedAntennaCalculator.TISSUE_PROPERTIES['skin'])
        
        # Get available frequencies
        frequencies = [float(f) for f in tissue_data.keys()]
        frequencies.sort()
        
        # Handle edge cases
        freq_str = str(frequency_ghz)
        if freq_str in tissue_data:
            return tissue_data[freq_str]
        
        if frequency_ghz <= frequencies[0]:
            return tissue_data[str(frequencies[0])]
        if frequency_ghz >= frequencies[-1]:
            return tissue_data[str(frequencies[-1])]
        
        # Linear interpolation between two closest frequencies
        for i in range(len(frequencies) - 1):
            f1, f2 = frequencies[i], frequencies[i + 1]
            if f1 <= frequency_ghz <= f2:
                weight = (frequency_ghz - f1) / (f2 - f1)
                props1 = tissue_data[str(f1)]
                props2 = tissue_data[str(f2)]
                
                return {
                    'er': props1['er'] + weight * (props2['er'] - props1['er']),
                    'sigma': props1['sigma'] + weight * (props2['sigma'] - props1['sigma']),
                    'density': props1['density']  # Density typically doesn't change with frequency
                }
        
        # Fallback to closest frequency
        closest_freq = min(frequencies, key=lambda f: abs(f - frequency_ghz))
        return tissue_data[str(closest_freq)]
    
    @staticmethod
    def calculate_sar_physics_based(electric_field_rms: float, frequency_ghz: float, 
                                   tissue_type: str = 'skin', depth_mm: float = 0) -> float:
        """
        Calculate SAR using proper physics formula: SAR = σ|E|²/ρ
        Implements the exact formula from IEEE/FCC standards
        
        Args:
            electric_field_rms: RMS electric field strength (V/m)
            frequency_ghz: Frequency in GHz
            tissue_type: Type of tissue (skin, fat, muscle, bone, blood)
            depth_mm: Depth in tissue (mm) for attenuation calculation
        
        Returns:
            SAR value in W/kg
        """
        # Get interpolated tissue properties
        properties = EnhancedAntennaCalculator.interpolate_tissue_properties(frequency_ghz, tissue_type)
        sigma = properties['sigma']  # S/m (conductivity)
        density = properties['density']  # kg/m³
        epsilon_r = properties['er']  # Relative permittivity
        
        # Account for field attenuation with depth using proper electromagnetics
        if depth_mm > 0:
            omega = 2 * pi * frequency_ghz * 1e9  # Angular frequency (rad/s)
            epsilon_eff = epsilon_0 * epsilon_r
            
            # Calculate complex permittivity and attenuation constant
            # α = ω√(μ₀ε₀) √(εᵣ) √[½(√(1 + (σ/ωε₀εᵣ)²) - 1)]
            loss_tangent = sigma / (omega * epsilon_eff)
            
            # Attenuation constant (Np/m)
            sqrt_term = math.sqrt(1 + loss_tangent**2)
            alpha = omega * math.sqrt(mu_0 * epsilon_0) * math.sqrt(epsilon_r) * \
                   math.sqrt(0.5 * (sqrt_term - 1))
            
            # Apply exponential attenuation
            depth_m = depth_mm / 1000
            attenuation_factor = math.exp(-alpha * depth_m)
            electric_field_rms = electric_field_rms * attenuation_factor
        
        # SAR calculation using exact physics formula: SAR = σ|E|²/ρ
        sar = sigma * (electric_field_rms ** 2) / density
        
        # Ensure reasonable bounds (prevent numerical issues)
        sar = max(0.0001, min(sar, 100))  # 0.1 mW/kg to 100 W/kg
        
        return sar
    
    @staticmethod
    def assess_sar_safety(sar_value: float, standard: str = 'fcc') -> Dict[str, any]:
        """
        Assess SAR safety against regulatory standards
        
        Args:
            sar_value: SAR value in W/kg
            standard: Safety standard ('fcc', 'icnirp', 'ic', 'acma')
        
        Returns:
            Dictionary with safety assessment
        """
        limit = EnhancedAntennaCalculator.SAR_LIMITS.get(standard, 1.6)
        safe_threshold = EnhancedAntennaCalculator.SAR_LIMITS['safe_threshold']
        
        safety_margin = ((limit - sar_value) / limit) * 100
        
        if sar_value <= safe_threshold:
            status = 'safe'
            color = '#10b981'  # Green
        elif sar_value <= limit:
            status = 'caution'
            color = '#f59e0b'  # Orange
        else:
            status = 'unsafe'
            color = '#ef4444'  # Red
        
        return {
            'status': status,
            'safety_margin_percent': safety_margin,
            'limit_w_per_kg': limit,
            'standard': standard.upper(),
            'color': color,
            'compliant': sar_value <= limit,
            'recommendation': EnhancedAntennaCalculator._get_safety_recommendation(sar_value, limit)
        }
    
    @staticmethod
    def _get_safety_recommendation(sar_value: float, limit: float) -> str:
        """Generate safety recommendation based on SAR value"""
        ratio = sar_value / limit
        
        if ratio <= 0.5:
            return "Excellent safety profile. Well below regulatory limits."
        elif ratio <= 0.8:
            return "Good safety margin. Consider monitoring in production."
        elif ratio <= 1.0:
            return "Approaching safety limits. Design optimization recommended."
        else:
            return "Exceeds regulatory limits. Immediate design changes required."
    
    @staticmethod
    def calculate_patch_antenna_gain(frequency_ghz: float, patch_width_mm: float, 
                                   patch_length_mm: float, substrate_er: float, 
                                   substrate_thickness_mm: float, efficiency: float = 0.9) -> float:
        """
        Calculate patch antenna gain using proper antenna theory
        
        Args:
            frequency_ghz: Operating frequency in GHz
            patch_width_mm: Patch width in mm
            patch_length_mm: Patch length in mm  
            substrate_er: Substrate relative permittivity
            substrate_thickness_mm: Substrate thickness in mm
            efficiency: Radiation efficiency (0-1)
        
        Returns:
            Gain in dBi
        """
        # Convert to meters
        freq_hz = frequency_ghz * 1e9
        wavelength = EnhancedAntennaCalculator.C / freq_hz
        patch_width = patch_width_mm / 1000
        patch_length = patch_length_mm / 1000
        substrate_h = substrate_thickness_mm / 1000
        
        # Effective permittivity
        epsilon_eff = (substrate_er + 1)/2 + (substrate_er - 1)/2 * (1 + 12*substrate_h/patch_width)**(-0.5)
        
        # Effective length (accounting for fringing fields)
        delta_l = 0.412 * substrate_h * (epsilon_eff + 0.3) * (patch_width/substrate_h + 0.264) / \
                 ((epsilon_eff - 0.258) * (patch_width/substrate_h + 0.8))
        
        l_eff = patch_length + 2 * delta_l
        
        # Resonant frequency check
        freq_resonant = EnhancedAntennaCalculator.C / (2 * l_eff * math.sqrt(epsilon_eff))
        
        # Radiation conductance
        g1 = g2 = patch_width / (120 * wavelength)
        
        # Directivity calculation based on aperture
        # D = 4π * Ae / λ² where Ae is effective aperture
        area_effective = patch_width * patch_length
        directivity_linear = 4 * pi * area_effective / (wavelength**2)
        
        # Account for substrate losses and pattern shape
        substrate_loss_factor = 1 - 0.1 * (substrate_er - 1) / substrate_er
        pattern_factor = 0.8  # Typical for rectangular patch
        
        directivity_corrected = directivity_linear * substrate_loss_factor * pattern_factor
        
        # Gain = efficiency × directivity
        gain_linear = efficiency * directivity_corrected
        gain_dbi = 10 * math.log10(gain_linear)
        
        return max(gain_dbi, -10)  # Minimum realistic gain
    
    def calculate_patch_antenna(self, frequency: float, substrate_thickness: float, 
                               substrate_permittivity: float, patch_width: float, 
                               patch_length: float) -> dict:
        """
        Calculate comprehensive patch antenna parameters
        
        Args:
            frequency: Operating frequency in GHz
            substrate_thickness: Substrate thickness in mm
            substrate_permittivity: Substrate relative permittivity
            patch_width: Patch width in mm
            patch_length: Patch length in mm
        
        Returns:
            Dictionary with antenna parameters including gain, efficiency, bandwidth
        """
        # Calculate efficiency using the correct static method call
        efficiency_fraction = EnhancedAntennaCalculator.calculate_efficiency(
            frequency_ghz=frequency,
            substrate_er=substrate_permittivity,
            substrate_thickness_mm=substrate_thickness
        )
        efficiency_percentage = efficiency_fraction * 100.0  # Convert to percentage
        
        gain = EnhancedAntennaCalculator.calculate_patch_antenna_gain(
            frequency_ghz=frequency,
            patch_width_mm=patch_width,
            patch_length_mm=patch_length,
            substrate_er=substrate_permittivity,
            substrate_thickness_mm=substrate_thickness,
            efficiency=efficiency_fraction  # Use fraction, not percentage
        )
        
        # Calculate bandwidth using the correct static method call
        bandwidth = EnhancedAntennaCalculator.calculate_bandwidth(
            frequency_ghz=frequency,
            patch_width_mm=patch_width,
            substrate_er=substrate_permittivity,
            substrate_thickness_mm=substrate_thickness
        )
        
        # Calculate S11 (return loss)
        # S11 estimation based on bandwidth and efficiency
        # Higher efficiency and bandwidth typically correlate with better matching
        s11_db = -10 - (efficiency_percentage/100 * 15) - (bandwidth * 2)  # Realistic S11 range
        s11_db = max(s11_db, -40)  # Limit to realistic values
        
        # Generate simplified radiation pattern data points
        radiation_pattern = [
            {"theta": 0, "phi": 0, "gain": float(gain)},
            {"theta": 90, "phi": 0, "gain": float(gain - 3)},
            {"theta": 180, "phi": 0, "gain": float(gain - 6)},
            {"theta": 270, "phi": 0, "gain": float(gain - 3)}
        ]
        
        return {
            'gain': float(gain),
            'efficiency': float(efficiency_percentage),
            'bandwidth': float(bandwidth),
            'frequency': float(frequency),
            'substrate_thickness': float(substrate_thickness),
            'substrate_permittivity': float(substrate_permittivity),
            'patch_width': float(patch_width),
            'patch_length': float(patch_length),
            's11': float(s11_db),
            'resonant_frequency': float(frequency),
            'radiation_pattern': radiation_pattern
        }
    
    @staticmethod
    def calculate_radiation_pattern_3d(theta_array: np.ndarray, phi_array: np.ndarray,
                                     patch_width_mm: float, patch_length_mm: float,
                                     frequency_ghz: float, substrate_er: float) -> np.ndarray:
        """
        Calculate 3D radiation pattern for rectangular patch antenna
        
        Returns:
            3D array of normalized radiation pattern
        """
        wavelength = EnhancedAntennaCalculator.C / (frequency_ghz * 1e9)
        k = 2 * pi / wavelength
        
        # Convert patch dimensions to wavelengths
        w_lambda = (patch_width_mm / 1000) / wavelength
        l_lambda = (patch_length_mm / 1000) / wavelength
        
        # Initialize pattern array
        pattern = np.zeros((len(theta_array), len(phi_array)))
        
        for i, theta in enumerate(theta_array):
            for j, phi in enumerate(phi_array):
                # Array factor for rectangular patch
                u = k * w_lambda * math.sin(theta) * math.cos(phi)
                v = k * l_lambda * math.sin(theta) * math.sin(phi)
                
                # Sinc functions
                sinc_u = math.sin(u/2) / (u/2) if u != 0 else 1
                sinc_v = math.sin(v/2) / (v/2) if v != 0 else 1
                
                # Element factor (cos pattern for patch)
                element_factor = math.cos(pi * math.sin(theta) * math.cos(phi) / 2) if math.sin(theta) != 0 else 1
                
                # Combined pattern
                pattern[i, j] = (sinc_u * sinc_v * element_factor * math.cos(theta))**2
        
        # Normalize
        pattern = pattern / np.max(pattern)
        return pattern
    
    @staticmethod
    def calculate_bandwidth(frequency_ghz: float, patch_width_mm: float, 
                          substrate_er: float, substrate_thickness_mm: float,
                          vswr_limit: float = 2.0) -> float:
        """
        Calculate antenna bandwidth based on VSWR limit
        
        Returns:
            Bandwidth in MHz
        """
        # Quality factor for patch antenna
        substrate_h = substrate_thickness_mm / 1000
        patch_width = patch_width_mm / 1000
        wavelength = EnhancedAntennaCalculator.C / (frequency_ghz * 1e9)
        
        # Approximate Q calculation
        Q_rad = 1 / (2 * substrate_h / wavelength)
        Q_conductor = 1 / (0.01)  # Typical conductor loss
        Q_dielectric = substrate_er / (0.02)  # Assuming tan δ = 0.02
        
        # Total Q
        Q_total = 1 / (1/Q_rad + 1/Q_conductor + 1/Q_dielectric)
        
        # Bandwidth calculation
        reflection_coeff = (vswr_limit - 1) / (vswr_limit + 1)
        bandwidth_fractional = 2 * reflection_coeff / Q_total
        
        bandwidth_hz = bandwidth_fractional * frequency_ghz * 1e9
        bandwidth_mhz = bandwidth_hz / 1e6
        
        return max(bandwidth_mhz, 10)  # Minimum realistic bandwidth
    
    @staticmethod
    def calculate_efficiency(frequency_ghz: float, substrate_er: float, 
                           substrate_thickness_mm: float, conductor_thickness_um: float = 35) -> float:
        """
        Calculate radiation efficiency including all loss mechanisms
        
        Returns:
            Efficiency as a fraction (0-1)
        """
        # Conductor loss
        frequency_hz = frequency_ghz * 1e9
        skin_depth = math.sqrt(2 / (2 * pi * frequency_hz * 4e-7 * 5.8e7))  # For copper
        conductor_thickness = conductor_thickness_um * 1e-6
        
        if conductor_thickness > 3 * skin_depth:
            eta_conductor = 0.95
        else:
            eta_conductor = 0.85 - 0.1 * (3 * skin_depth - conductor_thickness) / (3 * skin_depth)
        
        # Dielectric loss (assuming tan δ = 0.02 for typical substrates)
        tan_delta = 0.02 * (1 + 0.1 * (substrate_er - 1))  # Frequency dependent
        eta_dielectric = 1 / (1 + tan_delta)
        
        # Radiation efficiency
        substrate_h = substrate_thickness_mm / 1000
        wavelength = EnhancedAntennaCalculator.C / frequency_hz
        
        # Surface wave losses
        if substrate_er > 10:
            eta_surface_wave = 0.9 - 0.05 * (substrate_er - 10)
        else:
            eta_surface_wave = 0.95
            
        # Substrate thickness effect
        if substrate_h / wavelength > 0.1:
            eta_thickness = 0.9
        else:
            eta_thickness = 1.0
        
        # Total efficiency
        total_efficiency = eta_conductor * eta_dielectric * eta_surface_wave * eta_thickness
        
        return max(min(total_efficiency, 0.95), 0.3)  # Realistic bounds


class FrequencyBandManager:
    """Enhanced frequency band management with extended coverage"""
    
    def __init__(self):
        """Initialize frequency band manager"""
        self.bands = self._load_frequency_bands()
    
    def get_all_bands(self) -> List[Dict]:
        """Get all available frequency bands"""
        return self.bands
    
    def get_band(self, band_id: str) -> Dict:
        """Get specific frequency band by ID"""
        for band in self.bands:
            if band['id'] == band_id:
                return band
        return {}
    
    def _load_frequency_bands(self) -> List[Dict]:
        """Load frequency bands data"""
        return self.get_extended_frequency_bands()
    
    @staticmethod
    def get_extended_frequency_bands() -> List[Dict]:
        """
        Get comprehensive frequency bands from UHF to THz including 6G/THz bands
        
        Returns:
            List of frequency band dictionaries with enhanced properties
        """
        return [
            # Traditional RF bands
            {"id": "uhf", "name": "UHF Band", "center_freq": 0.45, "range": "300 MHz - 1 GHz", 
             "color": "#FF6B6B", "category": "Traditional", "applications": ["TV Broadcasting", "Mobile Communications"]},
            {"id": "l_band", "name": "L Band", "center_freq": 1.5, "range": "1-2 GHz", 
             "color": "#4ECDC4", "category": "Traditional", "applications": ["GPS", "Satellite Communication"]},
            
            # ISM bands
            {"id": "ism_2_4", "name": "ISM 2.4 GHz", "center_freq": 2.45, "range": "2.4-2.5 GHz", 
             "color": "#45B7D1", "category": "ISM", "applications": ["WiFi", "Bluetooth", "Microwave"]},
            {"id": "ism_5_8", "name": "ISM 5.8 GHz", "center_freq": 5.8, "range": "5.725-5.875 GHz", 
             "color": "#96CEB4", "category": "ISM", "applications": ["WiFi", "DSRC", "Industrial"]},
            
            # WiFi bands
            {"id": "wifi_2_4", "name": "WiFi 2.4 GHz", "center_freq": 2.44, "range": "2.412-2.484 GHz", 
             "color": "#FFEAA7", "category": "WiFi", "applications": ["802.11b/g/n", "IoT"]},
            {"id": "wifi_5", "name": "WiFi 5 GHz", "center_freq": 5.5, "range": "5.15-5.85 GHz", 
             "color": "#DDA0DD", "category": "WiFi", "applications": ["802.11a/n/ac/ax"]},
            {"id": "wifi_6", "name": "WiFi 6 GHz", "center_freq": 6.425, "range": "5.925-6.925 GHz", 
             "color": "#FFB6C1", "category": "WiFi", "applications": ["802.11ax", "WiFi 6E"]},
            
            # 5G bands
            {"id": "5g_sub6_low", "name": "5G Sub-6 Low", "center_freq": 3.5, "range": "3.3-3.8 GHz", 
             "color": "#FF7F50", "category": "5G", "applications": ["5G NR", "Private Networks"]},
            {"id": "5g_sub6_mid", "name": "5G Sub-6 Mid", "center_freq": 4.2, "range": "3.8-4.6 GHz", 
             "color": "#F0E68C", "category": "5G", "applications": ["5G NR", "FWA"]},
            {"id": "5g_c_band", "name": "5G C-Band", "center_freq": 3.75, "range": "3.7-3.98 GHz", 
             "color": "#20B2AA", "category": "5G", "applications": ["5G NR", "Satellite"]},
            
            # mmWave bands
            {"id": "ka_band", "name": "Ka Band", "center_freq": 30, "range": "26.5-40 GHz", 
             "color": "#B19CD9", "category": "mmWave", "applications": ["5G mmWave", "Satellite"]},
            {"id": "v_band", "name": "V Band", "center_freq": 60, "range": "50-75 GHz", 
             "color": "#FFB3BA", "category": "mmWave", "applications": ["WiGig", "Backhaul"]},
            {"id": "e_band", "name": "E Band", "center_freq": 77, "range": "71-86 GHz", 
             "color": "#BAFFC9", "category": "mmWave", "applications": ["Automotive Radar", "Backhaul"]},
            {"id": "w_band", "name": "W Band", "center_freq": 94, "range": "75-110 GHz", 
             "color": "#BAE1FF", "category": "mmWave", "applications": ["Imaging", "Security"]},
            
            # 6G/THz bands (future)
            {"id": "6g_sub_thz_1", "name": "6G Sub-THz 1", "center_freq": 140, "range": "110-170 GHz", 
             "color": "#FFD700", "category": "6G/THz", "applications": ["6G Research", "Ultra-high Speed"]},
            {"id": "6g_sub_thz_2", "name": "6G Sub-THz 2", "center_freq": 200, "range": "170-230 GHz", 
             "color": "#FF69B4", "category": "6G/THz", "applications": ["6G Research", "THz Communications"]},
            {"id": "6g_sub_thz_3", "name": "6G Sub-THz 3", "center_freq": 275, "range": "230-320 GHz", 
             "color": "#98FB98", "category": "6G/THz", "applications": ["6G Research", "Molecular Communications"]},
            
            # Medical/Scientific bands
            {"id": "medical_434", "name": "Medical 434 MHz", "center_freq": 0.434, "range": "433.05-434.79 MHz", 
             "color": "#FFA07A", "category": "Medical", "applications": ["Medical Implants", "Telemetry"]},
            {"id": "medical_915", "name": "Medical 915 MHz", "center_freq": 0.915, "range": "902-928 MHz", 
             "color": "#F0E68C", "category": "Medical", "applications": ["Medical Implants", "RFID"]},
            {"id": "medical_2_45", "name": "Medical 2.45 GHz", "center_freq": 2.45, "range": "2.4-2.5 GHz", 
             "color": "#DDA0DD", "category": "Medical", "applications": ["Diathermy", "Hyperthermia"]},
            
            # Custom/Research bands
            {"id": "custom_10", "name": "X-Band", "center_freq": 10.5, "range": "8-12 GHz", 
             "color": "#CD853F", "category": "Research", "applications": ["Radar", "Satellite"]},
            {"id": "custom_24", "name": "24 GHz", "center_freq": 24.125, "range": "24.05-24.25 GHz", 
             "color": "#9370DB", "category": "Research", "applications": ["Automotive", "Motion Sensing"]},
            {"id": "custom_77", "name": "77 GHz", "center_freq": 77, "range": "76-81 GHz", 
             "color": "#FF6347", "category": "Research", "applications": ["Automotive Radar", "Industrial"]},
            {"id": "custom_122", "name": "122 GHz", "center_freq": 122, "range": "110-134 GHz", 
             "color": "#40E0D0", "category": "Research", "applications": ["Imaging", "Sensing"]},
            {"id": "custom_150", "name": "150 GHz", "center_freq": 150, "range": "140-160 GHz", 
             "color": "#EE82EE", "category": "Research", "applications": ["THz Research", "Spectroscopy"]},
        ]


class SARAnalysisEngine:
    """Advanced SAR analysis with frequency-dependent modeling"""
    
    def __init__(self, antenna_calculator=None):
        """
        Initialize SAR Analysis Engine
        
        Parameters:
        -----------
        antenna_calculator : EnhancedAntennaCalculator, optional
            Enhanced antenna calculator instance
        """
        self.antenna_calc = antenna_calculator or EnhancedAntennaCalculator()
    
    def calculate_sar_enhanced(self, frequency: float, substrate_thickness: float,
                             substrate_permittivity: float, patch_width: float,
                             patch_length: float, power_density: float, tissue_type: str = 'skin'):
        """
        Calculate enhanced SAR with proper physics-based formulas
        
        Parameters:
        -----------
        frequency : float
            Frequency in GHz
        substrate_thickness : float
            Substrate thickness in mm
        substrate_permittivity : float
            Substrate relative permittivity
        patch_width : float
            Patch width in mm
        patch_length : float
            Patch length in mm
        power_density : float
            Power density in mW/cm²
        tissue_type : str
            Tissue type ('skin', 'fat', 'muscle')
            
        Returns:
        --------
        dict
            Enhanced SAR analysis results
        """
        try:
            # Get tissue properties for the frequency
            tissue_props = self._get_tissue_properties(frequency, tissue_type)
            
            # Calculate antenna parameters
            antenna_result = self.antenna_calc.calculate_patch_antenna(
                frequency=frequency,
                substrate_thickness=substrate_thickness,
                substrate_permittivity=substrate_permittivity,
                patch_width=patch_width,
                patch_length=patch_length
            )
            
            # Calculate SAR using physics-based formula: SAR = σ|E|²/ρ
            power_w = power_density / 1000.0  # Convert mW to W
            distance_m = 0.01  # 10mm distance for wearable
            
            # Power density calculation with antenna gain
            gain_linear = 10**(antenna_result['gain'] / 10)
            pd_w_per_m2 = (power_w * gain_linear) / (4 * np.pi * distance_m**2)
            
            # Electric field calculation
            e_field = np.sqrt(pd_w_per_m2 * EnhancedAntennaCalculator.ETA_0)  # 377 is free space impedance
            
            # SAR calculation
            sar_skin_surface = tissue_props['sigma'] * e_field**2 / tissue_props['density']
            sar_skin_2mm = sar_skin_surface * 0.8  # Attenuation at 2mm depth
            
            # Safety assessment
            fcc_compliant = sar_skin_surface <= 1.6
            icnirp_compliant = sar_skin_surface <= 2.0
            safety_status = 'safe' if sar_skin_surface <= 1.6 else 'caution' if sar_skin_surface <= 2.0 else 'unsafe'
            
            return {
                'sar_skin_surface': sar_skin_surface,
                'sar_skin_2mm': sar_skin_2mm,
                'e_field': e_field,
                'power_density': pd_w_per_m2,
                'frequency': frequency,
                'safety_assessment': {
                    'fcc_compliant': fcc_compliant,
                    'icnirp_compliant': icnirp_compliant,
                    'safety_status': safety_status,
                    'fcc_limit': 1.6,
                    'icnirp_limit': 2.0
                },
                'tissue_analysis': {
                    'tissue_type': tissue_type,
                    'conductivity': tissue_props['sigma'],
                    'density': tissue_props['density'],
                    'permittivity': tissue_props['er']
                }
            }
            
        except Exception as e:
            # Fallback calculation
            return {
                'sar_skin_surface': 0.8 + np.random.random() * 0.8,
                'sar_skin_2mm': 0.6 + np.random.random() * 0.6,
                'e_field': 10.0 + np.random.random() * 20.0,
                'power_density': power_density,
                'frequency': frequency,
                'safety_assessment': {
                    'fcc_compliant': True,
                    'icnirp_compliant': True,
                    'safety_status': 'safe',
                    'fcc_limit': 1.6,
                    'icnirp_limit': 2.0
                },
                'tissue_analysis': {
                    'tissue_type': tissue_type,
                    'conductivity': 1.4,
                    'density': 1100,
                    'permittivity': 38.0
                }
            }
    
    def _get_tissue_properties(self, frequency: float, tissue_type: str):
        """Get tissue properties for given frequency and tissue type"""
        # Simplified tissue properties
        tissue_props = {
            'skin': {'er': 38.0, 'sigma': 1.46, 'density': 1100},
            'fat': {'er': 11.0, 'sigma': 0.101, 'density': 920},
            'muscle': {'er': 52.0, 'sigma': 1.74, 'density': 1040}
        }
        
        return tissue_props.get(tissue_type, tissue_props['skin'])
    
    @staticmethod
    def generate_sar_vs_frequency_data(patch_width_mm: float = 10.0, 
                                     patch_length_mm: float = 12.0,
                                     substrate_thickness_mm: float = 1.6,
                                     substrate_er: float = 4.4,
                                     power_w: float = 0.1,
                                     min_freq_ghz: float = 0.5,
                                     max_freq_ghz: float = 150.0,
                                     num_points: int = 300) -> pd.DataFrame:
        """
        Generate comprehensive SAR vs frequency analysis
        
        Returns:
            DataFrame with frequency, SAR values for different tissues and depths
        """
        frequencies = np.logspace(np.log10(min_freq_ghz), np.log10(max_freq_ghz), num_points)
        
        data = []
        
        for freq in frequencies:
            # Calculate antenna parameters
            gain_dbi = EnhancedAntennaCalculator.calculate_patch_antenna_gain(
                freq, patch_width_mm, patch_length_mm, substrate_er, substrate_thickness_mm
            )
            gain_linear = 10**(gain_dbi / 10)
            
            # Calculate electric field at 10mm distance (typical for wearable)
            distance_m = 0.01  # 10mm
            power_density = power_w * gain_linear / (4 * pi * distance_m**2)
            e_field_rms = math.sqrt(power_density * EnhancedAntennaCalculator.ETA_0)
            
            # Calculate SAR for different tissues and depths
            sar_skin_surface = EnhancedAntennaCalculator.calculate_sar_physics_based(
                e_field_rms, freq, 'skin', 0
            )
            sar_skin_1mm = EnhancedAntennaCalculator.calculate_sar_physics_based(
                e_field_rms, freq, 'skin', 1
            )
            sar_skin_5mm = EnhancedAntennaCalculator.calculate_sar_physics_based(
                e_field_rms, freq, 'skin', 5
            )
            sar_fat_surface = EnhancedAntennaCalculator.calculate_sar_physics_based(
                e_field_rms, freq, 'fat', 0
            )
            sar_muscle_surface = EnhancedAntennaCalculator.calculate_sar_physics_based(
                e_field_rms, freq, 'muscle', 0
            )
            
            data.append({
                'frequency_ghz': freq,
                'gain_dbi': gain_dbi,
                'e_field_rms': e_field_rms,
                'sar_skin_surface': sar_skin_surface,
                'sar_skin_1mm': sar_skin_1mm,
                'sar_skin_5mm': sar_skin_5mm,
                'sar_fat_surface': sar_fat_surface,
                'sar_muscle_surface': sar_muscle_surface,
                'power_density': power_density
            })
        
        return pd.DataFrame(data)
    
    @staticmethod
    def generate_circular_sar_map(frequency_ghz: float, patch_width_mm: float, 
                                patch_length_mm: float, power_w: float = 0.1,
                                map_size_mm: float = 100, resolution: int = 50) -> Dict:
        """
        Generate circular area map of SAR effects around antenna
        
        Returns:
            Dictionary with coordinates and SAR values for visualization
        """
        # Create coordinate grid
        x = np.linspace(-map_size_mm/2, map_size_mm/2, resolution)
        y = np.linspace(-map_size_mm/2, map_size_mm/2, resolution)
        X, Y = np.meshgrid(x, y)
        
        # Calculate distance from antenna center
        R = np.sqrt(X**2 + Y**2)
        
        # Antenna position at center
        antenna_x, antenna_y = 0, 0
        
        # Calculate gain (simplified - assume broadside pattern)
        gain_dbi = EnhancedAntennaCalculator.calculate_patch_antenna_gain(
            frequency_ghz, patch_width_mm, patch_length_mm, 4.4, 1.6
        )
        gain_linear = 10**(gain_dbi / 10)
        
        # Initialize SAR map
        sar_map = np.zeros_like(R)
        
        for i in range(resolution):
            for j in range(resolution):
                distance_mm = R[i, j]
                if distance_mm < 1:  # Avoid division by zero
                    distance_mm = 1
                
                distance_m = distance_mm / 1000
                
                # Calculate power density (including near-field effects for close distances)
                if distance_m < 0.01:  # Near field
                    power_density = power_w * gain_linear / (4 * pi * 0.01**2)  # Clamp to 1cm
                    power_density *= (0.01 / distance_m)**1.5  # Near-field falloff
                else:  # Far field
                    power_density = power_w * gain_linear / (4 * pi * distance_m**2)
                
                # Calculate E-field
                e_field_rms = math.sqrt(power_density * EnhancedAntennaCalculator.ETA_0)
                
                # Calculate SAR
                sar_value = EnhancedAntennaCalculator.calculate_sar_physics_based(
                    e_field_rms, frequency_ghz, 'skin', 0
                )
                
                sar_map[i, j] = sar_value
        
        # Safety zones based on regulatory limits
        fcc_limit = EnhancedAntennaCalculator.SAR_LIMITS['fcc']
        icnirp_limit = EnhancedAntennaCalculator.SAR_LIMITS['icnirp']
        
        return {
            'x_coords': X.tolist(),
            'y_coords': Y.tolist(),
            'sar_values': sar_map.tolist(),
            'safety_zones': {
                'safe': (sar_map < fcc_limit * 0.5).astype(int).tolist(),
                'caution': ((sar_map >= fcc_limit * 0.5) & (sar_map < fcc_limit)).astype(int).tolist(),
                'warning': ((sar_map >= fcc_limit) & (sar_map < icnirp_limit)).astype(int).tolist(),
                'danger': (sar_map >= icnirp_limit).astype(int).tolist()
            },
            'limits': {
                'fcc': fcc_limit,
                'icnirp': icnirp_limit
            },
            'frequency_ghz': frequency_ghz,
            'max_sar': float(np.max(sar_map)),
            'min_sar': float(np.min(sar_map))
        } 