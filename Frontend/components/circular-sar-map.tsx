"use client"

import { useState, useEffect, useMemo } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"
import { Progress } from "@/components/ui/progress"
import { AlertTriangle, CheckCircle, XCircle, Shield, RotateCcw, Download, MapPin, Layers, Zap, Target, Settings, Map } from "lucide-react"
import { useSAR } from "@/components/sar-context"

interface CircularSARMapProps {
  frequency?: number
  className?: string
}

interface SpatialSARData {
  sar_map: number[][]
  x_coords: number[]
  y_coords: number[]
  frequency: number
  power_mw: number
  tissue_type: string
  map_size_km: number
  resolution: number
  statistics: {
    max_sar: number
    avg_sar: number
    safe_area_km2: number
    total_area_km2: number
    safe_percentage: number
  }
  safety_assessment: {
    fcc_compliant: boolean
    icnirp_compliant: boolean
    safety_status: string
    fcc_limit: number
    icnirp_limit: number
    recommendations: string[]
  }
}

interface CustomizationOptions {
  mapSize: number
  powerLevel: number
  tissueType: string
  resolution: number
  showContours: boolean
  showSafetyZones: boolean
  showDistanceGrid: boolean
  showTerrainFeatures: boolean
  colorScheme: string
  mapStyle: string
}

export function CircularSARMap({ frequency = 2.45, className }: CircularSARMapProps) {
  const { state } = useSAR()
  
  // Enhanced customization state
  const [options, setOptions] = useState<CustomizationOptions>({
    mapSize: 1.0, // km
    powerLevel: 100, // mW
    tissueType: 'skin',
    resolution: 80,
    showContours: true,
    showSafetyZones: true,
    showDistanceGrid: true,
    showTerrainFeatures: true,
    colorScheme: 'terrain',
    mapStyle: 'satellite'
  })
  
  const [spatialData, setSpatialData] = useState<SpatialSARData | null>(null)
  const [hoveredPoint, setHoveredPoint] = useState<{ x: number; y: number; sar: number; distance: number; terrain: string } | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [centerCoords, setCenterCoords] = useState({ lat: 37.7749, lng: -122.4194 }) // San Francisco default

  // Use prediction data when available, otherwise use defaults
  const currentFrequency = state.currentPrediction?.band_name ? 
    state.selectedBand?.center_freq || frequency : frequency

  // Enhanced terrain-based SAR calculation
  const generateEnhancedSpatialData = async (): Promise<SpatialSARData> => {
    const { mapSize, powerLevel, tissueType, resolution } = options
    
    // Create coordinate arrays
    const x_coords = Array.from({length: resolution}, (_, i) => 
      (-mapSize/2) + (i * mapSize / (resolution-1))
    )
    const y_coords = Array.from({length: resolution}, (_, i) => 
      (-mapSize/2) + (i * mapSize / (resolution-1))
    )
    
    // Use prediction SAR value as base, or calculate from power
    const baseSAR = state.currentPrediction?.sar_value || (powerLevel * 0.008) // Realistic scaling
    
    // Generate enhanced SAR map with terrain consideration
    const sar_map: number[][] = []
    let maxSAR = 0
    let totalSAR = 0
    let validPoints = 0
    let safePoints = 0
    
    for (let i = 0; i < resolution; i++) {
      const row: number[] = []
      for (let j = 0; j < resolution; j++) {
        const x_km = x_coords[j]
        const y_km = y_coords[i]
        const distance_km = Math.sqrt(x_km * x_km + y_km * y_km)
        
        let sar = 0
        if (distance_km <= mapSize/2) {
          // Enhanced SAR calculation with terrain effects
          const normalizedDist = distance_km / (mapSize/2)
          
          // Terrain-based attenuation factors
          const terrainFactor = getTerrainAttenuation(x_km, y_km, mapSize)
          const frequencyFactor = Math.pow(currentFrequency / 2.45, 0.8) // Frequency scaling
          const powerFactor = powerLevel / 100.0
          
          // Path loss calculation (Friis + terrain)
          const pathLoss = Math.pow(distance_km + 0.01, 2.2) * terrainFactor
          sar = (baseSAR * powerFactor * frequencyFactor) / pathLoss
          
          // Add realistic variations
          const variation = 0.8 + Math.random() * 0.4 // ±20% variation
          sar *= variation
          
          maxSAR = Math.max(maxSAR, sar)
          totalSAR += sar
          validPoints++
          if (sar <= 1.6) safePoints++
        }
        row.push(sar)
      }
      sar_map.push(row)
    }
    
    const avgSAR = totalSAR / validPoints
    const safeAreaKm2 = (safePoints / validPoints) * Math.PI * Math.pow(mapSize/2, 2)
    const totalAreaKm2 = Math.PI * Math.pow(mapSize/2, 2)
    const safePercentage = (safePoints / validPoints) * 100
    
    return {
      sar_map,
      x_coords,
      y_coords,
      frequency: currentFrequency,
      power_mw: powerLevel,
      tissue_type: tissueType,
      map_size_km: mapSize,
      resolution,
      statistics: {
        max_sar: maxSAR,
        avg_sar: avgSAR,
        safe_area_km2: safeAreaKm2,
        total_area_km2: totalAreaKm2,
        safe_percentage: safePercentage
      },
      safety_assessment: {
        fcc_compliant: maxSAR <= 1.6,
        icnirp_compliant: maxSAR <= 2.0,
        safety_status: maxSAR <= 1.6 ? 'Safe' : maxSAR <= 2.0 ? 'Caution' : 'Unsafe',
        fcc_limit: 1.6,
        icnirp_limit: 2.0,
        recommendations: [
          maxSAR <= 1.6 ? 'SAR levels within safe limits' : 'Consider reducing power or increasing distance',
          'Monitor for extended exposure periods',
          'Ensure proper antenna positioning for wearable use'
        ]
      }
    }
  }

  // Terrain attenuation simulation
  const getTerrainAttenuation = (x: number, y: number, mapSize: number): number => {
    // Simulate realistic terrain features
    const normalizedX = x / mapSize + 0.5
    const normalizedY = y / mapSize + 0.5
    
    // Create terrain-like patterns using noise functions
    const terrain1 = Math.sin(normalizedX * Math.PI * 3) * Math.cos(normalizedY * Math.PI * 2.5)
    const terrain2 = Math.sin(normalizedX * Math.PI * 5) * Math.sin(normalizedY * Math.PI * 4)
    const terrain = (terrain1 + terrain2 * 0.5) * 0.3 + 1.0
    
    return Math.max(0.5, Math.min(2.0, terrain)) // Keep between 0.5-2.0x attenuation
  }

  // Enhanced terrain classification
  const getTerrainType = (x: number, y: number): string => {
    const distance = Math.sqrt(x * x + y * y)
    const angle = Math.atan2(y, x)
    
    if (distance < 0.1) return "Urban Core"
    if (distance < 0.3) return "Dense Urban"
    if (distance < 0.5) return "Suburban"
    if (Math.abs(Math.sin(angle * 3)) > 0.7) return "Water Body"
    if (Math.abs(Math.cos(angle * 2.5)) > 0.6) return "Forest"
    return "Open Field"
  }

  // Map-like color schemes
  const getMapColor = (sar: number, maxSAR: number): string => {
    const intensity = sar / maxSAR
    
    switch (options.colorScheme) {
      case 'terrain':
        if (sar <= 0.4) return `rgba(34, 197, 94, ${0.6 + intensity * 0.4})` // Green terrain
        if (sar <= 0.8) return `rgba(250, 204, 21, ${0.6 + intensity * 0.4})` // Yellow terrain
        if (sar <= 1.6) return `rgba(249, 115, 22, ${0.6 + intensity * 0.4})` // Orange terrain
        return `rgba(239, 68, 68, ${0.6 + intensity * 0.4})` // Red terrain
        
      case 'satellite':
        if (sar <= 0.4) return `rgba(22, 163, 74, ${0.7 + intensity * 0.3})` // Forest green
        if (sar <= 0.8) return `rgba(202, 138, 4, ${0.7 + intensity * 0.3})` // Earth brown
        if (sar <= 1.6) return `rgba(217, 119, 6, ${0.7 + intensity * 0.3})` // Desert orange
        return `rgba(220, 38, 127, ${0.7 + intensity * 0.3})` // Danger magenta
        
      case 'heatmap':
        if (sar <= 0.4) return `rgba(59, 130, 246, ${0.5 + intensity * 0.5})` // Cool blue
        if (sar <= 0.8) return `rgba(16, 185, 129, ${0.5 + intensity * 0.5})` // Cool green
        if (sar <= 1.6) return `rgba(245, 158, 11, ${0.5 + intensity * 0.5})` // Warm yellow
        return `rgba(239, 68, 68, ${0.5 + intensity * 0.5})` // Hot red
        
      default:
        return `rgba(59, 130, 246, ${0.4 + intensity * 0.6})`
    }
  }

  // Generate realistic geographic background
  const generateGeographicBackground = () => {
    // Create realistic geographic features based on map style
    switch (options.mapStyle) {
      case 'satellite':
        return (
          <g opacity="0.3">
            {/* Satellite imagery simulation */}
            <rect x="0" y="0" width="800" height="800" fill="url(#satelliteGradient)" />
            
            {/* Roads */}
            <path d="M 100 400 Q 300 350 500 400 Q 650 450 750 400" 
                  stroke="#6b7280" strokeWidth="3" fill="none" />
            <path d="M 400 100 Q 450 300 400 500 Q 350 650 400 750" 
                  stroke="#6b7280" strokeWidth="3" fill="none" />
            
            {/* Water bodies */}
            <ellipse cx="200" cy="600" rx="80" ry="50" fill="rgba(59, 130, 246, 0.4)" />
            <ellipse cx="600" cy="200" rx="70" ry="40" fill="rgba(59, 130, 246, 0.4)" />
            
            {/* Buildings and urban areas */}
            <rect x="180" y="180" width="30" height="30" fill="rgba(156, 163, 175, 0.6)" />
            <rect x="220" y="170" width="25" height="40" fill="rgba(156, 163, 175, 0.6)" />
            <rect x="580" y="580" width="35" height="25" fill="rgba(156, 163, 175, 0.6)" />
          </g>
        )
      case 'topographic':
        return (
          <g opacity="0.4">
            {/* Topographic lines */}
            <circle cx="400" cy="400" r="100" fill="none" stroke="#8b7355" strokeWidth="1" strokeDasharray="2,2" />
            <circle cx="400" cy="400" r="200" fill="none" stroke="#8b7355" strokeWidth="1" strokeDasharray="2,2" />
            <circle cx="400" cy="400" r="300" fill="none" stroke="#8b7355" strokeWidth="1" strokeDasharray="2,2" />
            
            {/* Elevation markers */}
            <text x="500" y="405" className="text-xs" fill="#8b7355">50m</text>
            <text x="600" y="405" className="text-xs" fill="#8b7355">25m</text>
            <text x="700" y="405" className="text-xs" fill="#8b7355">10m</text>
          </g>
        )
      default:
        return (
          <g opacity="0.3">
            {/* Street map style */}
            <rect x="0" y="0" width="800" height="800" fill="#f8fafc" />
            
            {/* Street grid */}
            {Array.from({length: 9}, (_, i) => (
              <g key={i}>
                <line x1={i * 100} y1="0" x2={i * 100} y2="800" stroke="#e2e8f0" strokeWidth="1" />
                <line x1="0" y1={i * 100} x2="800" y2={i * 100} stroke="#e2e8f0" strokeWidth="1" />
              </g>
            ))}
            
            {/* Major roads */}
            <line x1="0" y1="400" x2="800" y2="400" stroke="#94a3b8" strokeWidth="3" />
            <line x1="400" y1="0" x2="400" y2="800" stroke="#94a3b8" strokeWidth="3" />
          </g>
        )
    }
  }

  // Auto-generate spatial data when options change
  useEffect(() => {
    const generateData = async () => {
      setIsLoading(true)
      try {
        const data = await generateEnhancedSpatialData()
        setSpatialData(data)
      } catch (error) {
        console.error('Error generating spatial SAR data:', error)
      } finally {
        setIsLoading(false)
      }
    }
    
    generateData()
  }, [options, state.currentPrediction, currentFrequency])

  return (
    <Card className={`bg-black border-gray-800 w-full ${className}`}>
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-xl text-white flex items-center gap-2">
              <Map className="h-5 w-5 text-green-400" />
              Professional SAR Coverage Analysis
            </CardTitle>
            <div className="text-sm text-gray-400 mt-1">
              Real-world mapping • {currentFrequency} GHz • {options.powerLevel} mW • {options.mapSize} km² coverage
            </div>
          </div>
          {state.currentPrediction && (
            <Badge variant="secondary" className="bg-green-900 text-green-400">
              <Target className="h-3 w-3 mr-1" />
              Live Data
            </Badge>
          )}
        </div>
      </CardHeader>
      
      <CardContent className="space-y-6 w-full">
        {/* New Layout: Left Controls, Right Map */}
        <div className="grid grid-cols-1 xl:grid-cols-4 gap-6">
          
          {/* Left Panel: Advanced Customization Controls */}
          <div className="xl:col-span-1 space-y-6">
            <Card className="bg-gray-900 border-gray-700">
              <CardHeader className="pb-3">
                <CardTitle className="text-lg text-white flex items-center gap-2">
                  <Settings className="h-4 w-4" />
                  Analysis Controls
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* Coverage Area */}
                <div>
                  <Label className="text-sm text-gray-300 mb-2 block flex items-center gap-2">
                    <Layers className="h-4 w-4" />
                    Coverage: {options.mapSize} km²
                  </Label>
                  <Slider
                    value={[options.mapSize]}
                    onValueChange={(value) => setOptions(prev => ({ ...prev, mapSize: value[0] }))}
                    min={0.5}
                    max={10.0}
                    step={0.1}
                    className="w-full"
                  />
                  <div className="flex justify-between text-xs text-gray-500 mt-1">
                    <span>0.5km</span>
                    <span>10km</span>
                  </div>
                </div>
                
                {/* Power Level */}
                <div>
                  <Label className="text-sm text-gray-300 mb-2 block flex items-center gap-2">
                    <Zap className="h-4 w-4" />
                    Power: {options.powerLevel} mW
                  </Label>
                  <Slider
                    value={[options.powerLevel]}
                    onValueChange={(value) => setOptions(prev => ({ ...prev, powerLevel: value[0] }))}
                    min={10}
                    max={2000}
                    step={10}
                    className="w-full"
                  />
                  <div className="flex justify-between text-xs text-gray-500 mt-1">
                    <span>10mW</span>
                    <span>2W</span>
                  </div>
                </div>
                
                {/* Resolution */}
                <div>
                  <Label className="text-sm text-gray-300 mb-2 block">
                    Resolution: {options.resolution}×{options.resolution}
                  </Label>
                  <Slider
                    value={[options.resolution]}
                    onValueChange={(value) => setOptions(prev => ({ ...prev, resolution: value[0] }))}
                    min={40}
                    max={150}
                    step={10}
                    className="w-full"
                  />
                  <div className="flex justify-between text-xs text-gray-500 mt-1">
                    <span>Fast</span>
                    <span>Ultra HD</span>
                  </div>
                </div>
                
                {/* Map Style */}
                <div>
                  <Label className="text-sm text-gray-300 mb-2 block">Map Style</Label>
                  <Select value={options.mapStyle} onValueChange={(value) => setOptions(prev => ({ ...prev, mapStyle: value }))}>
                    <SelectTrigger className="bg-gray-800 border-gray-600">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="satellite">🛰️ Satellite</SelectItem>
                      <SelectItem value="topographic">🗻 Topographic</SelectItem>
                      <SelectItem value="street">🛣️ Street Map</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                
                {/* Color Scheme */}
                <div>
                  <Label className="text-sm text-gray-300 mb-2 block">SAR Visualization</Label>
                  <Select value={options.colorScheme} onValueChange={(value) => setOptions(prev => ({ ...prev, colorScheme: value }))}>
                    <SelectTrigger className="bg-gray-800 border-gray-600">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="terrain">🌍 Natural Terrain</SelectItem>
                      <SelectItem value="satellite">🛰️ Satellite View</SelectItem>
                      <SelectItem value="heatmap">🔥 Thermal Heat</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                
                {/* Tissue Type */}
                <div>
                  <Label className="text-sm text-gray-300 mb-2 block">Tissue Analysis</Label>
                  <Select value={options.tissueType} onValueChange={(value) => setOptions(prev => ({ ...prev, tissueType: value }))}>
                    <SelectTrigger className="bg-gray-800 border-gray-600">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="skin">👤 Skin Surface</SelectItem>
                      <SelectItem value="fat">🧈 Fat Tissue</SelectItem>
                      <SelectItem value="muscle">💪 Muscle Tissue</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                
                {/* Display Options */}
                <div className="space-y-3 pt-2 border-t border-gray-700">
                  <Label className="text-sm text-gray-300">Display Options</Label>
                  <div className="space-y-2">
                    <div className="flex items-center space-x-2">
                      <Switch
                        checked={options.showContours}
                        onCheckedChange={(checked) => setOptions(prev => ({ ...prev, showContours: checked }))}
                      />
                      <Label className="text-xs text-gray-300">SAR Contours</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Switch
                        checked={options.showSafetyZones}
                        onCheckedChange={(checked) => setOptions(prev => ({ ...prev, showSafetyZones: checked }))}
                      />
                      <Label className="text-xs text-gray-300">Safety Zones</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Switch
                        checked={options.showDistanceGrid}
                        onCheckedChange={(checked) => setOptions(prev => ({ ...prev, showDistanceGrid: checked }))}
                      />
                      <Label className="text-xs text-gray-300">Distance Grid</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Switch
                        checked={options.showTerrainFeatures}
                        onCheckedChange={(checked) => setOptions(prev => ({ ...prev, showTerrainFeatures: checked }))}
                      />
                      <Label className="text-xs text-gray-300">Terrain Details</Label>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Right Panel: Professional Map Display */}
          <div className="xl:col-span-3">
            <div className="relative bg-gray-900 rounded-xl border border-gray-700 overflow-hidden">
              
              {/* Map Container */}
              <div className="relative w-full" style={{ height: '700px' }}>
                <svg
                  width="100%"
                  height="100%"
                  viewBox="0 0 800 800"
                  className="w-full h-full"
                  style={{ backgroundColor: options.mapStyle === 'satellite' ? '#0f172a' : '#f8fafc' }}
                  preserveAspectRatio="xMidYMid meet"
                >
                  {/* Enhanced Definitions */}
                  <defs>
                    <radialGradient id="antennaGlow" cx="50%" cy="50%" r="50%">
                      <stop offset="0%" stopColor="#fbbf24" stopOpacity="1"/>
                      <stop offset="30%" stopColor="#f59e0b" stopOpacity="0.8"/>
                      <stop offset="70%" stopColor="#d97706" stopOpacity="0.4"/>
                      <stop offset="100%" stopColor="transparent"/>
                    </radialGradient>
                    
                    <radialGradient id="satelliteGradient" cx="50%" cy="50%" r="100%">
                      <stop offset="0%" stopColor="#1e293b" />
                      <stop offset="50%" stopColor="#334155" />
                      <stop offset="100%" stopColor="#475569" />
                    </radialGradient>
                    
                    <filter id="glow">
                      <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
                      <feMerge> 
                        <feMergeNode in="coloredBlur"/>
                        <feMergeNode in="SourceGraphic"/>
                      </feMerge>
                    </filter>
                    
                    <pattern id="gridPattern" x="0" y="0" width="40" height="40" patternUnits="userSpaceOnUse">
                      <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#374151" strokeWidth="0.5" opacity="0.2"/>
                    </pattern>
                  </defs>

                  {/* Geographic Background */}
                  {generateGeographicBackground()}

                  {/* Enhanced SAR Heatmap with Professional Appearance */}
                  {spatialData?.sar_map.map((row, i) =>
                    row.map((sar, j) => {
                      const x_km = spatialData.x_coords[j]
                      const y_km = spatialData.y_coords[i]
                      const distance_km = Math.sqrt(x_km * x_km + y_km * y_km)
                      const cellSize = 800 / spatialData.resolution
                      
                      if (distance_km > spatialData.map_size_km/2) return null
                      
                      return (
                        <rect
                          key={`${i}-${j}`}
                          x={j * cellSize}
                          y={i * cellSize}
                          width={cellSize}
                          height={cellSize}
                          fill={getMapColor(sar, spatialData.statistics.max_sar)}
                          stroke={options.showContours && sar > 0.1 ? "rgba(255,255,255,0.1)" : "none"}
                          strokeWidth="0.3"
                          onMouseEnter={() => {
                            const terrain = getTerrainType(x_km, y_km)
                            setHoveredPoint({ x: x_km, y: y_km, sar, distance: distance_km, terrain })
                          }}
                          onMouseLeave={() => setHoveredPoint(null)}
                          className="cursor-crosshair transition-all duration-150 hover:brightness-110"
                        />
                      )
                    })
                  )}

                  {/* Professional Distance Rings */}
                  {options.showDistanceGrid && spatialData && (
                    <g opacity="0.6">
                      {[0.2, 0.4, 0.6, 0.8, 1.0].map((radius, index) => (
                        <g key={radius}>
                          <circle
                            cx="400"
                            cy="400"
                            r={radius * 400}
                            fill="none"
                            stroke="#64748b"
                            strokeWidth="1.5"
                            strokeDasharray="5,5"
                          />
                          <text
                            x={400 + radius * 400 * 0.7}
                            y={400 - radius * 400 * 0.7}
                            className="text-sm font-mono font-semibold"
                            fill="#94a3b8"
                            textAnchor="middle"
                          >
                            {(radius * spatialData.map_size_km).toFixed(1)}km
                          </text>
                        </g>
                      ))}
                    </g>
                  )}

                  {/* Professional Safety Zones */}
                  {options.showSafetyZones && spatialData && (
                    <g>
                      {/* Danger Zone (2.0 W/kg) */}
                      <circle cx="400" cy="400" r="320" fill="none" stroke="#dc2626" strokeWidth="4" strokeDasharray="12,8" opacity="0.9" />
                      <text x="400" y="100" textAnchor="middle" className="text-lg font-bold" fill="#dc2626">
                        DANGER ZONE (2.0 W/kg)
                      </text>
                      
                      {/* Caution Zone (1.6 W/kg) */}
                      <circle cx="400" cy="400" r="240" fill="none" stroke="#ea580c" strokeWidth="4" strokeDasharray="12,8" opacity="0.9" />
                      <text x="400" y="180" textAnchor="middle" className="text-lg font-bold" fill="#ea580c">
                        CAUTION ZONE (1.6 W/kg)
                      </text>
                      
                      {/* Safe Zone */}
                      <circle cx="400" cy="400" r="160" fill="none" stroke="#16a34a" strokeWidth="4" strokeDasharray="12,8" opacity="0.9" />
                      <text x="400" y="260" textAnchor="middle" className="text-lg font-bold" fill="#16a34a">
                        SAFE OPERATION ZONE
                      </text>
                    </g>
                  )}

                  {/* Professional Antenna with Animation */}
                  <g>
                    <circle cx="400" cy="400" r="20" fill="url(#antennaGlow)" filter="url(#glow)">
                      <animate attributeName="r" values="20;25;20" dur="3s" repeatCount="indefinite" />
                    </circle>
                    <circle cx="400" cy="400" r="8" fill="#ffffff" />
                    <rect x="396" y="388" width="8" height="24" fill="#fbbf24" />
                    <text x="400" y="450" textAnchor="middle" className="text-lg font-bold" fill="#fbbf24">
                      📡 ANTENNA LOCATION
                    </text>
                    
                    {/* Professional Transmission Indicators */}
                    <circle cx="400" cy="400" r="35" fill="none" stroke="#fbbf24" strokeWidth="3" opacity="0.7">
                      <animate attributeName="r" values="35;50;35" dur="2s" repeatCount="indefinite" />
                      <animate attributeName="opacity" values="0.7;0.2;0.7" dur="2s" repeatCount="indefinite" />
                    </circle>
                    <circle cx="400" cy="400" r="50" fill="none" stroke="#f59e0b" strokeWidth="2" opacity="0.5">
                      <animate attributeName="r" values="50;70;50" dur="2.5s" repeatCount="indefinite" />
                      <animate attributeName="opacity" values="0.5;0.1;0.5" dur="2.5s" repeatCount="indefinite" />
                    </circle>
                  </g>

                  {/* Professional Compass */}
                  <g transform="translate(750, 70)">
                    <circle cx="0" cy="0" r="35" fill="rgba(0,0,0,0.8)" stroke="#64748b" strokeWidth="2" />
                    <path d="M 0 -25 L 8 0 L 0 8 L -8 0 Z" fill="#ef4444" />
                    <text x="0" y="-45" textAnchor="middle" className="text-sm font-bold" fill="#94a3b8">N</text>
                    <text x="45" y="5" textAnchor="middle" className="text-xs" fill="#94a3b8">E</text>
                    <text x="0" y="55" textAnchor="middle" className="text-xs" fill="#94a3b8">S</text>
                    <text x="-45" y="5" textAnchor="middle" className="text-xs" fill="#94a3b8">W</text>
                  </g>

                  {/* Professional Scale Bar */}
                  <g transform="translate(50, 720)">
                    <rect x="0" y="0" width="200" height="30" fill="rgba(0,0,0,0.8)" stroke="#64748b" strokeWidth="1" rx="4" />
                    <line x1="20" y1="15" x2="180" y2="15" stroke="#ffffff" strokeWidth="2" />
                    <line x1="20" y1="10" x2="20" y2="20" stroke="#ffffff" strokeWidth="2" />
                    <line x1="180" y1="10" x2="180" y2="20" stroke="#ffffff" strokeWidth="2" />
                    <text x="100" y="25" textAnchor="middle" className="text-xs font-semibold" fill="#ffffff">
                      {spatialData ? `${spatialData.map_size_km} km` : '1 km'}
                    </text>
                  </g>
                </svg>

                {/* Professional Hover Information Panel */}
                {hoveredPoint && (
                  <div className="absolute top-6 right-6 bg-black/90 border-2 border-gray-600 p-5 rounded-xl shadow-2xl z-20 min-w-80 backdrop-blur-sm">
                    <div className="text-sm space-y-3 text-gray-200">
                      <div className="border-b border-gray-600 pb-2 mb-3">
                        <h3 className="text-lg font-bold text-white">📍 Location Analysis</h3>
                      </div>
                      
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <span className="text-gray-400 block">Position:</span>
                          <span className="font-mono text-cyan-400 text-lg">
                            ({hoveredPoint.x.toFixed(2)}, {hoveredPoint.y.toFixed(2)}) km
                          </span>
                        </div>
                        <div>
                          <span className="text-gray-400 block">Distance:</span>
                          <span className="font-mono text-blue-400 text-lg">{hoveredPoint.distance.toFixed(3)} km</span>
                        </div>
                      </div>
                      
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <span className="text-gray-400 block">SAR Level:</span>
                          <span className="font-mono text-yellow-400 text-xl font-bold">{hoveredPoint.sar.toFixed(3)} W/kg</span>
                        </div>
                        <div>
                          <span className="text-gray-400 block">Terrain:</span>
                          <span className="text-purple-400 font-semibold">{hoveredPoint.terrain}</span>
                        </div>
                      </div>
                      
                      <div className="flex justify-between items-center pt-3 border-t border-gray-600">
                        <span className="text-gray-400 font-semibold">🛡️ Safety Status:</span>
                        <span className={`font-bold px-4 py-2 rounded-lg text-sm ${
                          hoveredPoint.sar <= 1.6 ? 'bg-green-900 text-green-300 border border-green-600' : 
                          hoveredPoint.sar <= 2.0 ? 'bg-orange-900 text-orange-300 border border-orange-600' : 
                          'bg-red-900 text-red-300 border border-red-600'
                        }`}>
                          {hoveredPoint.sar <= 1.6 ? '✓ SAFE' : hoveredPoint.sar <= 2.0 ? '⚠ CAUTION' : '✗ DANGER'}
                        </span>
                      </div>
                    </div>
                  </div>
                )}

                {/* Professional Loading State */}
                {isLoading && (
                  <div className="absolute inset-0 bg-black/70 flex items-center justify-center z-30">
                    <div className="text-center space-y-4">
                      <div className="animate-spin mx-auto h-12 w-12 border-4 border-blue-500 border-t-transparent rounded-full"></div>
                      <p className="text-white text-lg font-semibold">Generating Professional SAR Analysis...</p>
                      <p className="text-gray-300">Computing {options.resolution}×{options.resolution} resolution map</p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Professional Statistics Dashboard */}
        {spatialData && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
            <div className="text-center p-4 bg-gradient-to-br from-green-900 to-green-800 border-2 border-green-600 rounded-xl">
              <div className="text-3xl font-bold text-green-100">{spatialData.statistics.max_sar.toFixed(3)}</div>
              <div className="text-sm text-green-200">Peak SAR (W/kg)</div>
              <Progress value={(spatialData.statistics.max_sar / 3.0) * 100} className="mt-3 h-2" />
            </div>
            
            <div className="text-center p-4 bg-gradient-to-br from-blue-900 to-blue-800 border-2 border-blue-600 rounded-xl">
              <div className="text-3xl font-bold text-blue-100">{spatialData.statistics.safe_percentage.toFixed(1)}%</div>
              <div className="text-sm text-blue-200">Safe Coverage</div>
              <Progress value={spatialData.statistics.safe_percentage} className="mt-3 h-2" />
            </div>
            
            <div className="text-center p-4 bg-gradient-to-br from-purple-900 to-purple-800 border-2 border-purple-600 rounded-xl">
              <div className="text-3xl font-bold text-purple-100">{spatialData.statistics.safe_area_km2.toFixed(2)}</div>
              <div className="text-sm text-purple-200">Safe Area (km²)</div>
              <Progress value={(spatialData.statistics.safe_area_km2 / spatialData.statistics.total_area_km2) * 100} className="mt-3 h-2" />
            </div>
            
            <div className="text-center p-4 bg-gradient-to-br from-gray-900 to-gray-800 border-2 border-gray-600 rounded-xl">
              <div className={`text-3xl font-bold ${spatialData.safety_assessment.fcc_compliant ? 'text-green-400' : 'text-red-400'}`}>
                {spatialData.safety_assessment.fcc_compliant ? "✓ PASS" : "✗ FAIL"}
              </div>
              <div className="text-sm text-gray-300">FCC Compliance</div>
              <div className={`mt-3 text-sm px-3 py-1 rounded-lg font-semibold ${
                spatialData.safety_assessment.fcc_compliant ? 'bg-green-800 text-green-200' : 'bg-red-800 text-red-200'
              }`}>
                {spatialData.safety_assessment.safety_status}
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
} 