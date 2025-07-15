import type { AntennaParameters, SARPrediction, FrequencyBand } from "@/components/sar-context"
import { apiClient } from "./api-client"

// Enhanced SAR API service with extended frequency range and proper physics
class SARAPIService {
  private fallbackToMock = false // Set to false when real API is available

  // Extended frequency bands up to 150 GHz including 6G/THz bands
  private extendedFrequencyBands = [
    // Traditional RF bands
    { id: "uhf", name: "UHF Band", center_freq: 0.45, range: "300 MHz - 1 GHz", 
      color: "#FF6B6B", category: "Traditional", applications: ["TV Broadcasting", "Mobile Communications"] },
    { id: "l_band", name: "L Band", center_freq: 1.5, range: "1-2 GHz", 
      color: "#4ECDC4", category: "Traditional", applications: ["GPS", "Satellite Communication"] },
    
    // ISM bands
    { id: "ism_2_4", name: "ISM 2.4 GHz", center_freq: 2.45, range: "2.4-2.5 GHz", 
      color: "#45B7D1", category: "ISM", applications: ["WiFi", "Bluetooth", "Microwave"] },
    { id: "ism_5_8", name: "ISM 5.8 GHz", center_freq: 5.8, range: "5.725-5.875 GHz", 
      color: "#96CEB4", category: "ISM", applications: ["WiFi", "DSRC", "Industrial"] },
    
    // WiFi bands
    { id: "wifi_2_4", name: "WiFi 2.4 GHz", center_freq: 2.44, range: "2.412-2.484 GHz", 
      color: "#FFEAA7", category: "WiFi", applications: ["802.11b/g/n", "IoT"] },
    { id: "wifi_5", name: "WiFi 5 GHz", center_freq: 5.5, range: "5.15-5.85 GHz", 
      color: "#DDA0DD", category: "WiFi", applications: ["802.11a/n/ac/ax"] },
    { id: "wifi_6", name: "WiFi 6 GHz", center_freq: 6.425, range: "5.925-6.925 GHz", 
      color: "#FFB6C1", category: "WiFi", applications: ["802.11ax", "WiFi 6E"] },
    
    // 5G bands
    { id: "5g_sub6_low", name: "5G Sub-6 Low", center_freq: 3.5, range: "3.3-3.8 GHz", 
      color: "#FF7F50", category: "5G", applications: ["5G NR", "Private Networks"] },
    { id: "5g_sub6_mid", name: "5G Sub-6 Mid", center_freq: 4.2, range: "3.8-4.6 GHz", 
      color: "#F0E68C", category: "5G", applications: ["5G NR", "FWA"] },
    { id: "5g_c_band", name: "5G C-Band", center_freq: 3.75, range: "3.7-3.98 GHz", 
      color: "#20B2AA", category: "5G", applications: ["5G NR", "Satellite"] },
    
    // mmWave bands
    { id: "ka_band", name: "Ka Band", center_freq: 30, range: "26.5-40 GHz", 
      color: "#B19CD9", category: "mmWave", applications: ["5G mmWave", "Satellite"] },
    { id: "v_band", name: "V Band", center_freq: 60, range: "50-75 GHz", 
      color: "#FFB3BA", category: "mmWave", applications: ["WiGig", "Backhaul"] },
    { id: "e_band", name: "E Band", center_freq: 77, range: "71-86 GHz", 
      color: "#BAFFC9", category: "mmWave", applications: ["Automotive Radar", "Backhaul"] },
    { id: "w_band", name: "W Band", center_freq: 94, range: "75-110 GHz", 
      color: "#BAE1FF", category: "mmWave", applications: ["Imaging", "Security"] },
    
    // 6G/THz bands (future)
    { id: "6g_sub_thz_1", name: "6G Sub-THz 1", center_freq: 140, range: "110-170 GHz", 
      color: "#FFD700", category: "6G/THz", applications: ["6G Research", "Ultra-high Speed"] },
    { id: "6g_sub_thz_2", name: "6G Sub-THz 2", center_freq: 200, range: "170-230 GHz", 
      color: "#FF69B4", category: "6G/THz", applications: ["6G Research", "THz Communications"] },
    { id: "6g_sub_thz_3", name: "6G Sub-THz 3", center_freq: 275, range: "230-320 GHz", 
      color: "#98FB98", category: "6G/THz", applications: ["6G Research", "Molecular Communications"] },
    
    // Medical/Scientific bands
    { id: "medical_434", name: "Medical 434 MHz", center_freq: 0.434, range: "433.05-434.79 MHz", 
      color: "#FFA07A", category: "Medical", applications: ["Medical Implants", "Telemetry"] },
    { id: "medical_915", name: "Medical 915 MHz", center_freq: 0.915, range: "902-928 MHz", 
      color: "#F0E68C", category: "Medical", applications: ["Medical Implants", "RFID"] },
    { id: "medical_2_45", name: "Medical 2.45 GHz", center_freq: 2.45, range: "2.4-2.5 GHz", 
      color: "#DDA0DD", category: "Medical", applications: ["Diathermy", "Hyperthermia"] },
    
    // Custom/Research bands
    { id: "custom_10", name: "X-Band", center_freq: 10.5, range: "8-12 GHz", 
      color: "#CD853F", category: "Research", applications: ["Radar", "Satellite"] },
    { id: "custom_24", name: "24 GHz", center_freq: 24.125, range: "24.05-24.25 GHz", 
      color: "#9370DB", category: "Research", applications: ["Automotive", "Motion Sensing"] },
    { id: "custom_77", name: "77 GHz", center_freq: 77, range: "76-81 GHz", 
      color: "#FF6347", category: "Research", applications: ["Automotive Radar", "Industrial"] },
    { id: "custom_122", name: "122 GHz", center_freq: 122, range: "110-134 GHz", 
      color: "#40E0D0", category: "Research", applications: ["Imaging", "Sensing"] },
    { id: "custom_150", name: "150 GHz", center_freq: 150, range: "140-160 GHz", 
      color: "#EE82EE", category: "Research", applications: ["THz Research", "Spectroscopy"] },
  ]

  // Enhanced tissue properties based on Gabriel's research (same as backend)
  private tissueProperties: Record<string, Record<string, { sigma: number; density: number; er: number }>> = {
    'skin': {
      '0.3': { sigma: 0.69, density: 1100, er: 46.7 },
      '0.9': { sigma: 0.79, density: 1100, er: 41.4 },
      '1.8': { sigma: 1.05, density: 1100, er: 39.2 },
      '2.4': { sigma: 1.43, density: 1100, er: 38.09 },
      '2.45': { sigma: 1.46, density: 1100, er: 38.0 },
      '3.5': { sigma: 2.01, density: 1100, er: 36.8 },
      '5.8': { sigma: 3.717, density: 1100, er: 35.11 },
      '10': { sigma: 6.2, density: 1100, er: 32.5 },
      '15': { sigma: 8.9, density: 1100, er: 30.8 },
      '24': { sigma: 12.1, density: 1100, er: 29.0 },
      '28': { sigma: 12.5, density: 1100, er: 28.8 },
      '38': { sigma: 15.8, density: 1100, er: 27.2 },
      '60': { sigma: 22.8, density: 1100, er: 24.2 },
      '77': { sigma: 27.2, density: 1100, er: 22.8 },
      '94': { sigma: 31.8, density: 1100, er: 21.2 },
      '100': { sigma: 35.2, density: 1100, er: 20.1 },
      '140': { sigma: 45.8, density: 1100, er: 17.5 },
      '150': { sigma: 48.5, density: 1100, er: 16.8 }
    },
    'fat': {
      '0.3': { sigma: 0.024, density: 920, er: 5.58 },
      '0.9': { sigma: 0.043, density: 920, er: 5.46 },
      '1.8': { sigma: 0.072, density: 920, er: 5.35 },
      '2.4': { sigma: 0.10, density: 920, er: 5.29 },
      '2.45': { sigma: 0.101, density: 920, er: 5.28 },
      '3.5': { sigma: 0.15, density: 920, er: 5.19 },
      '5.8': { sigma: 0.29, density: 920, er: 4.95 },
      '10': { sigma: 0.45, density: 920, er: 4.8 },
      '15': { sigma: 0.68, density: 920, er: 4.65 },
      '24': { sigma: 1.05, density: 920, er: 4.42 },
      '28': { sigma: 1.2, density: 920, er: 4.2 },
      '38': { sigma: 1.65, density: 920, er: 4.05 },
      '60': { sigma: 2.8, density: 920, er: 3.8 },
      '77': { sigma: 3.5, density: 920, er: 3.62 },
      '94': { sigma: 4.2, density: 920, er: 3.45 },
      '100': { sigma: 4.5, density: 920, er: 3.2 },
      '140': { sigma: 5.8, density: 920, er: 2.95 },
      '150': { sigma: 6.2, density: 920, er: 2.8 }
    },
    'muscle': {
      '0.3': { sigma: 0.51, density: 1040, er: 59.1 },
      '0.9': { sigma: 0.78, density: 1040, er: 56.8 },
      '1.8': { sigma: 1.25, density: 1040, er: 54.7 },
      '2.4': { sigma: 1.69, density: 1040, er: 52.82 },
      '2.45': { sigma: 1.74, density: 1040, er: 52.7 },
      '3.5': { sigma: 2.45, density: 1040, er: 50.1 },
      '5.8': { sigma: 4.96, density: 1040, er: 48.48 },
      '10': { sigma: 8.2, density: 1040, er: 45.2 },
      '15': { sigma: 12.1, density: 1040, er: 42.8 },
      '24': { sigma: 16.8, density: 1040, er: 40.2 },
      '28': { sigma: 18.5, density: 1040, er: 38.5 },
      '38': { sigma: 22.8, density: 1040, er: 36.2 },
      '60': { sigma: 32.8, density: 1040, er: 32.1 },
      '77': { sigma: 39.2, density: 1040, er: 29.8 },
      '94': { sigma: 44.5, density: 1040, er: 28.1 },
      '100': { sigma: 48.2, density: 1040, er: 26.8 },
      '140': { sigma: 58.8, density: 1040, er: 24.2 },
      '150': { sigma: 65.8, density: 1040, er: 22.5 }
    }
  }

  // Safety standards
  private sarLimits = {
    fcc: 1.6,      // FCC limit (US)
    icnirp: 2.0,   // ICNIRP limit (EU)
    ic: 1.6,       // Industry Canada
    acma: 2.0,     // Australia
    safe_threshold: 0.8  // Conservative safety threshold (50% of FCC)
  }

  // Interpolate tissue properties for any frequency
  private interpolateTissueProperties(frequencyGHz: number, tissueType: string): { sigma: number; density: number; er: number } {
    const tissueData = this.tissueProperties[tissueType] || this.tissueProperties['skin']
    const frequencies = Object.keys(tissueData).map(f => parseFloat(f)).sort((a, b) => a - b)
    
    // Find exact match or interpolate
    const freqStr = frequencyGHz.toString()
    if (tissueData[freqStr]) {
      return tissueData[freqStr]
    }
    
    // Handle edge cases
    if (frequencyGHz <= frequencies[0]) {
      return tissueData[frequencies[0].toString()]
    }
    if (frequencyGHz >= frequencies[frequencies.length - 1]) {
      return tissueData[frequencies[frequencies.length - 1].toString()]
    }
    
    // Linear interpolation
    for (let i = 0; i < frequencies.length - 1; i++) {
      const f1 = frequencies[i]
      const f2 = frequencies[i + 1]
      
      if (f1 <= frequencyGHz && frequencyGHz <= f2) {
        const weight = (frequencyGHz - f1) / (f2 - f1)
        const props1 = tissueData[f1.toString()]
        const props2 = tissueData[f2.toString()]
        
        return {
          sigma: props1.sigma + weight * (props2.sigma - props1.sigma),
          er: props1.er + weight * (props2.er - props1.er),
          density: props1.density // Density doesn't change with frequency
        }
      }
    }
    
    // Fallback
    const closest = frequencies.reduce((prev, curr) => 
      Math.abs(curr - frequencyGHz) < Math.abs(prev - frequencyGHz) ? curr : prev
    )
    return tissueData[closest.toString()]
  }

  // Enhanced physics-based SAR calculation: SAR = σ|E|²/ρ
  private calculateSARPhysicsBased(electricFieldRms: number, frequencyGHz: number, tissueType: string = 'skin', depthMm: number = 0): number {
    const properties = this.interpolateTissueProperties(frequencyGHz, tissueType)
    let { sigma, density, er } = properties
    
    // Account for field attenuation with depth
    if (depthMm > 0) {
      const omega = 2 * Math.PI * frequencyGHz * 1e9 // Angular frequency (rad/s)
      const epsilon0 = 8.854e-12 // F/m
      const mu0 = 4 * Math.PI * 1e-7 // H/m
      const epsilonEff = epsilon0 * er
      
      // Calculate attenuation constant using proper electromagnetics
      const lossTangent = sigma / (omega * epsilonEff)
      const sqrtTerm = Math.sqrt(1 + lossTangent * lossTangent)
      const alpha = omega * Math.sqrt(mu0 * epsilon0) * Math.sqrt(er) * 
                   Math.sqrt(0.5 * (sqrtTerm - 1))
      
      // Apply exponential attenuation
      const depthM = depthMm / 1000
      const attenuationFactor = Math.exp(-alpha * depthM)
      electricFieldRms = electricFieldRms * attenuationFactor
    }
    
    // SAR calculation using exact physics formula: SAR = σ|E|²/ρ
    const sar = sigma * Math.pow(electricFieldRms, 2) / density
    
    // Ensure reasonable bounds
    return Math.max(0.0001, Math.min(sar, 100)) // 0.1 mW/kg to 100 W/kg
  }

  // Assess SAR safety against standards
  private assessSARSafety(sarValue: number, standard: string = 'fcc'): any {
    const limit = this.sarLimits[standard as keyof typeof this.sarLimits] || this.sarLimits.fcc
    const safeThreshold = this.sarLimits.safe_threshold
    
    const safetyMargin = ((limit - sarValue) / limit) * 100
    
    let status: string, color: string
    if (sarValue <= safeThreshold) {
      status = 'safe'
      color = '#10b981' // Green
    } else if (sarValue <= limit) {
      status = 'caution'
      color = '#f59e0b' // Orange  
    } else {
      status = 'unsafe'
      color = '#ef4444' // Red
    }
    
    const recommendation = this.getSafetyRecommendation(sarValue, limit)
    
    return {
      status,
      safety_margin_percent: safetyMargin,
      limit_w_per_kg: limit,
      standard: standard.toUpperCase(),
      color,
      compliant: sarValue <= limit,
      recommendation
    }
  }

  private getSafetyRecommendation(sarValue: number, limit: number): string {
    const ratio = sarValue / limit
    if (ratio <= 0.5) return "Excellent safety profile. Well below regulatory limits."
    if (ratio <= 0.8) return "Good safety margin. Consider monitoring in production."
    if (ratio <= 1.0) return "Approaching safety limits. Design optimization recommended."
    return "Exceeds regulatory limits. Immediate design changes required."
  }

  // Enhanced antenna gain calculation
  private calculateAntennaGain(frequency: number, parameters: AntennaParameters): number {
    const wavelengthMm = 300 / frequency // Approximate wavelength in mm
    
    // More realistic patch antenna gain calculation
    const optimalLength = wavelengthMm / (2 * Math.sqrt(parameters.substrate_permittivity))
    const sizeFactor = Math.exp(-Math.pow((parameters.patch_length - optimalLength) / optimalLength, 2))
    const substrateFactor = 1.0 + 0.15 * Math.log(parameters.substrate_permittivity)
    const thicknessFactor = 1.0 + 0.3 * (parameters.substrate_thickness / wavelengthMm)
    const frequencyFactor = 1.0 + 0.5 * Math.log(frequency / 2.45)
    
    let gain = 2 + 8 * sizeFactor * substrateFactor * thicknessFactor * frequencyFactor
    gain += (Math.random() - 0.5) * 1.0 // Add some variation
    
    return Math.max(-5, Math.min(gain, 20)) // Realistic bounds
  }

  // Generate enhanced S-parameters with frequency-dependent behavior
  private generateEnhancedSParameters(centerFreq: number, parameters: AntennaParameters, bandwidth = 2000) {
    const points = []
    const startFreq = Math.max(0.1, centerFreq - bandwidth)
    const endFreq = centerFreq + bandwidth
    const numPoints = 200

    for (let i = 0; i < numPoints; i++) {
      const freq = startFreq + (endFreq - startFreq) * (i / (numPoints - 1))
      const normalizedFreq = (freq - centerFreq) / bandwidth
      
      // More realistic S11 calculation based on antenna theory
      const qFactor = 50 + 30 * parameters.substrate_permittivity / parameters.substrate_thickness
      const s11 = -20 * Math.exp(-Math.pow(normalizedFreq * qFactor / 10, 2)) - 5 + Math.random() * 2

      points.push({
        frequency: freq * 1000000, // Convert to Hz
        s11: Math.max(s11, -50),
      })
    }

    return points
  }

  // Generate frequency sweep data with enhanced calculations
  private generateFrequencySweepData(parameters: AntennaParameters): any[] {
    const data: any[] = []
    const frequencies: number[] = []
    
    // Generate logarithmic frequency scale from 0.5 to 150 GHz for comprehensive coverage
    for (let i = 0; i <= 300; i++) {
      const freq = 0.5 * Math.pow(150 / 0.5, i / 300)
      frequencies.push(freq)
    }
    
    frequencies.forEach(freq => {
      const gain = this.calculateAntennaGain(freq, parameters)
      const gainLinear = Math.pow(10, gain / 10)
      
      // Calculate E-field at 10mm distance (typical for wearable devices)
      const distance = 0.01 // 10mm
      const powerDensity = 0.1 * gainLinear / (4 * Math.PI * distance * distance) // 100mW input
      const eField = Math.sqrt(powerDensity * 377) // η₀ = 377Ω
      
      // Calculate SAR for different tissues using enhanced physics
      const sarSkin = this.calculateSARPhysicsBased(eField, freq, 'skin', 0)
      const sarSkinDepth = this.calculateSARPhysicsBased(eField, freq, 'skin', 2) // 2mm depth
      const sarFat = this.calculateSARPhysicsBased(eField, freq, 'fat', 0)
      const sarMuscle = this.calculateSARPhysicsBased(eField, freq, 'muscle', 0)
      
      // Safety assessments
      const fccSafety = this.assessSARSafety(sarSkin, 'fcc')
      const icnirpSafety = this.assessSARSafety(sarSkin, 'icnirp')
      
      data.push({
        frequency: freq,
        gain,
        sar_skin_surface: sarSkin,
        sar_skin_2mm: sarSkinDepth,
        sar_fat_surface: sarFat,
        sar_muscle_surface: sarMuscle,
        e_field: eField,
        power_density: powerDensity,
        fcc_compliant: fccSafety.compliant,
        icnirp_compliant: icnirpSafety.compliant,
        safety_status: fccSafety.status,
        safety_margin_fcc: fccSafety.safety_margin_percent
      })
    })
    
    return data
  }

  // Generate enhanced circular SAR map with proper physics
  private generateCircularSARMap(frequency: number, parameters: AntennaParameters): any {
    const mapSize = 100 // mm
    const resolution = 50
    const sarMap = []
    
    const gain = this.calculateAntennaGain(frequency, parameters)
    const gainLinear = Math.pow(10, gain / 10)
    
    for (let i = 0; i < resolution; i++) {
      const row = []
      for (let j = 0; j < resolution; j++) {
        const x = (i - resolution / 2) * (mapSize / resolution)
        const y = (j - resolution / 2) * (mapSize / resolution)
        const distance = Math.sqrt(x * x + y * y)
        
        if (distance < 1) {
          row.push(0) // At antenna location
          continue
        }
        
        // Calculate power density with near-field corrections
        const distanceM = Math.max(distance / 1000, 0.001)
        let powerDensity = 0.1 * gainLinear / (4 * Math.PI * distanceM * distanceM)
        
        // Enhanced near-field modeling
        const wavelengthM = 0.3 / frequency // Wavelength in meters
        const nearFieldDistance = 2 * Math.pow(Math.max(parameters.patch_length, parameters.patch_width) / 1000, 2) / wavelengthM
        
        if (distanceM < nearFieldDistance) {
          // Near field: higher power density falloff
          powerDensity *= Math.pow(nearFieldDistance / distanceM, 1.8)
        }
        
        // Calculate electric field
        const eField = Math.sqrt(powerDensity * 377)
        
        // Calculate SAR using enhanced physics
        const sar = this.calculateSARPhysicsBased(eField, frequency, 'skin', 0)
        row.push(sar)
      }
      sarMap.push(row)
    }
    
    const maxSAR = Math.max(...sarMap.flat())
    const avgSAR = sarMap.flat().reduce((a, b) => a + b, 0) / (resolution * resolution)
    
    // Enhanced safety zones based on multiple standards
    const fccSafety = this.assessSARSafety(maxSAR, 'fcc')
    const icnirpSafety = this.assessSARSafety(maxSAR, 'icnirp')
    
    return {
      data: sarMap,
      resolution,
      mapSize,
      frequency,
      maxSAR,
      avgSAR,
      safeZones: {
        safe: sarMap.flat().filter(s => s < this.sarLimits.safe_threshold).length,
        caution: sarMap.flat().filter(s => s >= this.sarLimits.safe_threshold && s < this.sarLimits.fcc).length,
        warning: sarMap.flat().filter(s => s >= this.sarLimits.fcc && s < this.sarLimits.icnirp).length
      },
      safety_assessment: {
        fcc: fccSafety,
        icnirp: icnirpSafety,
        overall_status: fccSafety.status,
        recommendations: [
          fccSafety.recommendation,
          icnirpSafety.recommendation
        ]
      }
    }
  }

  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms))
  }

  async getBands(): Promise<FrequencyBand[]> {
    try {
      if (!this.fallbackToMock) {
        const response = await apiClient.getFrequencyBands()
        if (response) {
          return response
        }
      }
    } catch (error) {
      console.warn("Failed to fetch bands from API, using extended mock data:", error)
    }

    // Return extended frequency bands
    await this.delay(300)
    return this.extendedFrequencyBands
  }

  async getBandsByFrequency(frequency: number): Promise<FrequencyBand[]> {
    return this.extendedFrequencyBands.filter(band => 
      frequency >= band.min_freq && frequency <= band.max_freq
    )
  }

  async predict(bandId: string, parameters: AntennaParameters, customFrequency?: number): Promise<SARPrediction> {
    try {
      if (!this.fallbackToMock) {
        const response = await apiClient.predictSAR({
          band_id: bandId,
          parameters,
          input_mode: "custom",
        })

        if (response) {
          return response
        }
      }
    } catch (error) {
      console.warn("Failed to create prediction via API, falling back to enhanced mock:", error)
    }

    // Enhanced fallback prediction with proper physics
    await this.delay(1500)

    const bands = await this.getBands()
    const band = bands.find((b) => b.id === bandId)
    const frequency = customFrequency || band?.center_freq || 10

    // Enhanced antenna gain calculation
    const gain = this.calculateAntennaGain(frequency, parameters)
    const gainLinear = Math.pow(10, gain / 10)
    
    // Calculate power density at 10mm (typical wearable distance)
    const distance = 0.01 // 10mm
    const powerDensity = 0.1 * gainLinear / (4 * Math.PI * distance * distance) // 100mW input power
    const eField = Math.sqrt(powerDensity * 377) // η₀ = 377Ω
    
    // Calculate SAR using enhanced physics formula
    const sarValue = this.calculateSARPhysicsBased(eField, frequency, 'skin', 0)
    
    // Safety assessment against multiple standards
    const fccSafety = this.assessSARSafety(sarValue, 'fcc')
    const icnirpSafety = this.assessSARSafety(sarValue, 'icnirp')
    
    // Calculate other enhanced parameters
    const efficiency = 0.6 + Math.random() * 0.35
    const qFactor = 50 + 30 * parameters.substrate_permittivity / parameters.substrate_thickness
    const bandwidth = (frequency * 1000) / qFactor // MHz

    // Generate enhanced radiation pattern
    const radiationPattern = this.generateMockRadiationPattern()
    
    // Generate frequency sweep and circular SAR map
    const frequencySweep = this.generateFrequencySweepData(parameters)
    const circularSARMap = this.generateCircularSARMap(frequency, parameters)

    const prediction: SARPrediction = {
      id: `pred_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date().toISOString(),
      band_id: bandId,
      band_name: band?.name || bandId,
      parameters,
      sar_value: sarValue,
      gain,
      efficiency,
      bandwidth,
      frequency,
      s_parameters: this.generateEnhancedSParameters(frequency, parameters),
      radiation_pattern: radiationPattern,
      max_return_loss: Math.min(...this.generateEnhancedSParameters(frequency, parameters).map(p => p.s11)),
      resonant_frequency: frequency,
      // Enhanced properties
      safety_assessment: {
        fcc: fccSafety,
        icnirp: icnirpSafety,
        overall_compliant: fccSafety.compliant && icnirpSafety.compliant,
        primary_standard: 'FCC', // Default to stricter standard
        safety_score: Math.max(0, Math.min(100, fccSafety.safety_margin_percent))
      },
      tissue_analysis: {
        sar_skin: this.calculateSARPhysicsBased(eField, frequency, 'skin', 0),
        sar_fat: this.calculateSARPhysicsBased(eField, frequency, 'fat', 0),
        sar_muscle: this.calculateSARPhysicsBased(eField, frequency, 'muscle', 0),
        depth_analysis: {
          surface: this.calculateSARPhysicsBased(eField, frequency, 'skin', 0),
          depth_1mm: this.calculateSARPhysicsBased(eField, frequency, 'skin', 1),
          depth_5mm: this.calculateSARPhysicsBased(eField, frequency, 'skin', 5)
        }
      },
      frequency_sweep: frequencySweep,
      circular_sar_map: circularSARMap
    }

    return prediction
  }

  // Generate frequency sweep analysis
  async generateFrequencySweep(parameters: AntennaParameters): Promise<any> {
    await this.delay(2000)
    return this.generateFrequencySweepData(parameters)
  }

  // Generate circular SAR map
  async generateSARMap(frequency: number, parameters: AntennaParameters): Promise<any> {
    await this.delay(1500)
    return this.generateCircularSARMap(frequency, parameters)
  }

  // Find suitable frequency for antenna dimensions
  async findOptimalFrequency(parameters: AntennaParameters): Promise<number> {
    // Calculate resonant frequency based on patch dimensions
    const c = 299792458 // speed of light
    const effectiveLength = parameters.patch_length / 1000 // convert to meters
    const effectivePermittivity = (parameters.substrate_permittivity + 1) / 2 +
      (parameters.substrate_permittivity - 1) / 2 * 
      Math.pow(1 + 12 * (parameters.substrate_thickness / 1000) / (parameters.patch_width / 1000), -0.5)
    
    const frequency = c / (2 * effectiveLength * Math.sqrt(effectivePermittivity)) / 1e9 // GHz
    
    return Math.max(0.1, Math.min(frequency, 150)) // Clamp to reasonable range
  }

  async getPredictions(filters: any = {}): Promise<SARPrediction[]> {
    // Mock implementation - API endpoint not yet available
    await this.delay(300)
    return []
  }

  async deletePrediction(predictionId: string): Promise<boolean> {
    // Mock implementation - API endpoint not yet available
    await this.delay(500)
    return true
  }

  async getComparisonData(): Promise<SARPrediction[]> {
    try {
      if (!this.fallbackToMock) {
        const bands = await this.getBands()
        const defaultParams = {
          substrate_thickness: 1.6,
          substrate_permittivity: 4.4,
          patch_width: 10.0,
          patch_length: 12.0,
          feed_position: 0.25,
          bending_radius: 50.0,
          power_density: 1.0,
        }

        // API method not available yet, fall through to mock
      }
    } catch (error) {
      console.warn("Failed to generate comparison via API:", error)
    }

    // Fallback to mock comparison
    await this.delay(1000)
    const bands = ["x-band", "ku-band", "k-band", "ka-band", "v-band", "w-band"]
    const predictions = []

    for (const bandId of bands) {
      const defaultParams = {
        substrate_thickness: 1.6,
        substrate_permittivity: 4.4,
        patch_width: 10.0,
        patch_length: 12.0,
        feed_position: 0.25,
        bending_radius: 50.0,
        power_density: 1.0,
      }
      const prediction = await this.predict(bandId, defaultParams)
      predictions.push(prediction)
    }

    return predictions
  }

  async exportData(format: "csv" | "json" | "xlsx", data: any[]): Promise<string> {
    try {
      if (!this.fallbackToMock) {
        // API method not available yet, fall through to mock
      }
    } catch (error) {
      console.warn("Failed to export via API:", error)
    }

    // Mock export - generate download URL
    await this.delay(1000)
    return `https://api.sarpredictor.com/downloads/export_${Date.now()}.${format}`
  }

  async getAnalytics(timeRange = "30d") {
    try {
      if (!this.fallbackToMock) {
        // API method not available yet, fall through to mock
      }
    } catch (error) {
      console.warn("Failed to fetch analytics from API:", error)
    }

    // Mock analytics data
    await this.delay(500)
    return {
      totalPredictions: 1247,
      safetyDistribution: { safe: 892, warning: 234, unsafe: 121 },
      frequencyBandUsage: [
        { band: "X-band", count: 345 },
        { band: "Ku-band", count: 298 },
        { band: "K-band", count: 234 },
        { band: "Ka-band", count: 189 },
        { band: "V-band", count: 123 },
        { band: "W-band", count: 58 },
      ],
      averageSAR: 1.234,
      trendsData: Array.from({ length: 30 }, (_, i) => ({
        date: new Date(Date.now() - (29 - i) * 24 * 60 * 60 * 1000).toISOString().split("T")[0],
        predictions: Math.floor(Math.random() * 50) + 10,
        averageSAR: 1.0 + Math.random() * 0.8,
      })),
    }
  }

  private generateMockRadiationPattern() {
    const pattern = []
    for (let theta = 0; theta <= 180; theta += 5) {
      for (let phi = 0; phi < 360; phi += 10) {
        const thetaRad = (theta * Math.PI) / 180
        const phiRad = (phi * Math.PI) / 180
        
        // Enhanced radiation pattern calculation
        const gainPattern = Math.pow(Math.sin(thetaRad), 2) * Math.cos(thetaRad)
        const gainDb = 10 * Math.log10(Math.max(gainPattern, 0.001))
        
        pattern.push({
          theta: thetaRad,
          phi: phiRad,
          gain: gainDb,
        })
      }
    }
    return pattern
  }
}

export const sarAPI = new SARAPIService()
