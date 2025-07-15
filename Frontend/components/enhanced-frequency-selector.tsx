"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Switch } from "@/components/ui/switch"
import { Slider } from "@/components/ui/slider"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Separator } from "@/components/ui/separator"
import { useSAR, type FrequencyBand } from "@/components/sar-context"
import { sarAPI } from "@/lib/sar-api"
import { 
  Zap, Radio, Satellite, Wifi, Heart, Brain, 
  Settings, Calculator, Search, Target, Waves, Cpu
} from "lucide-react"

interface EnhancedFrequencySelectorProps {
  onFrequencyChange?: (frequency: number, band: FrequencyBand | null) => void
  onBandChange?: (band: FrequencyBand) => void
}

export function EnhancedFrequencySelector({ 
  onFrequencyChange, 
  onBandChange 
}: EnhancedFrequencySelectorProps) {
  const { state, dispatch } = useSAR()
  const [customFrequency, setCustomFrequency] = useState<number>(10.0)
  const [useCustomFrequency, setUseCustomFrequency] = useState(false)
  const [frequencyBands, setFrequencyBands] = useState<FrequencyBand[]>([])
  const [filteredBands, setFilteredBands] = useState<FrequencyBand[]>([])
  const [searchTerm, setSearchTerm] = useState("")
  const [selectedCategory, setSelectedCategory] = useState<string>("all")
  const [optimalFrequency, setOptimalFrequency] = useState<number | null>(null)

  // Load frequency bands
  useEffect(() => {
    const loadBands = async () => {
      try {
        const bands = await sarAPI.getBands()
        setFrequencyBands(bands)
        setFilteredBands(bands)
      } catch (error) {
        console.error("Failed to load frequency bands:", error)
      }
    }
    loadBands()
  }, [])

  // Filter bands based on search and category
  useEffect(() => {
    let filtered = frequencyBands

    // Category filter
    if (selectedCategory !== "all") {
      filtered = filtered.filter(band => {
        const bandName = band.name.toLowerCase()
        const bandId = band.id.toLowerCase()
        switch (selectedCategory) {
          case "traditional":
            return ["uhf", "l-band", "s-band", "c-band", "x-band", "ku-band", "k-band", "ka-band"].includes(band.id)
          case "5g6g":
            return band.id.includes("5g") || band.id.includes("6g") || band.center_freq >= 24
          case "ism":
            return band.id.includes("ism")
          case "wifi":
            return band.id.includes("wifi")
          case "medical":
            return band.id.includes("medical") || band.id.includes("mics") || band.id.includes("wmts")
          case "mmwave":
            return band.center_freq >= 30
          case "thz":
            return band.center_freq >= 95
          default:
            return true
        }
      })
    }

    // Search filter
    if (searchTerm) {
      filtered = filtered.filter(band => 
        band.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        band.range.toLowerCase().includes(searchTerm.toLowerCase()) ||
        band.id.toLowerCase().includes(searchTerm.toLowerCase())
      )
    }

    setFilteredBands(filtered)
  }, [frequencyBands, searchTerm, selectedCategory])

  // Calculate optimal frequency for current antenna parameters
  useEffect(() => {
    const calculateOptimal = async () => {
      if (state.parameters) {
        try {
          const optimal = await sarAPI.findOptimalFrequency(state.parameters)
          setOptimalFrequency(optimal)
        } catch (error) {
          console.error("Failed to calculate optimal frequency:", error)
        }
      }
    }
    calculateOptimal()
  }, [state.parameters])

  const handleBandSelect = (bandId: string) => {
    const band = frequencyBands.find(b => b.id === bandId)
    if (band) {
      dispatch({ type: "SET_BAND", payload: band })
      onBandChange?.(band)
      
      if (!useCustomFrequency) {
        onFrequencyChange?.(band.center_freq, band)
      }
    }
  }

  const handleCustomFrequencyChange = (freq: number) => {
    setCustomFrequency(freq)
    if (useCustomFrequency) {
      onFrequencyChange?.(freq, null)
      
      // Find bands that include this frequency
      sarAPI.getBandsByFrequency(freq).then(bands => {
        if (bands.length > 0) {
          const primaryBand = bands[0]
          dispatch({ type: "SET_BAND", payload: primaryBand })
        }
      })
    }
  }

  const handleUseCustomToggle = (checked: boolean) => {
    setUseCustomFrequency(checked)
    if (checked) {
      onFrequencyChange?.(customFrequency, null)
    } else if (state.selectedBand) {
      onFrequencyChange?.(state.selectedBand.center_freq, state.selectedBand)
    }
  }

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case "traditional": return <Radio className="h-4 w-4" />
      case "5g6g": return <Zap className="h-4 w-4" />
      case "ism": return <Settings className="h-4 w-4" />
      case "wifi": return <Wifi className="h-4 w-4" />
      case "medical": return <Heart className="h-4 w-4" />
      case "mmwave": return <Waves className="h-4 w-4" />
      case "thz": return <Satellite className="h-4 w-4" />
      default: return <Target className="h-4 w-4" />
    }
  }

  const getCategoryColor = (category: string) => {
    switch (category) {
      case "traditional": return "bg-blue-50 text-blue-700 border-blue-200"
      case "5g6g": return "bg-green-50 text-green-700 border-green-200"
      case "ism": return "bg-purple-50 text-purple-700 border-purple-200"
      case "wifi": return "bg-cyan-50 text-cyan-700 border-cyan-200"
      case "medical": return "bg-red-50 text-red-700 border-red-200"
      case "mmwave": return "bg-orange-50 text-orange-700 border-orange-200"
      case "thz": return "bg-indigo-50 text-indigo-700 border-indigo-200"
      default: return "bg-gray-50 text-gray-700 border-gray-200"
    }
  }

  const formatFrequency = (freq: number) => {
    if (freq >= 1000) {
      return `${(freq / 1000).toFixed(1)} THz`
    } else if (freq >= 1) {
      return `${freq.toFixed(1)} GHz`
    } else {
      return `${(freq * 1000).toFixed(0)} MHz`
    }
  }

  const getApplications = (band: FrequencyBand) => {
    const apps: string[] = []
    
    if (band.id.includes("ism") || band.id.includes("medical")) {
      apps.push("Medical")
    }
    if (band.id.includes("wifi")) {
      apps.push("WiFi")
    }
    if (band.id.includes("5g") || band.id.includes("6g")) {
      apps.push("Cellular")
    }
    if (band.center_freq >= 30) {
      apps.push("mmWave")
    }
    if (band.center_freq >= 95) {
      apps.push("6G/THz")
    }
    
    return apps
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Calculator className="h-5 w-5" />
          Enhanced Frequency Selection
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Custom Frequency Toggle */}
        <div className="flex items-center justify-between">
          <div className="space-y-1">
            <Label className="text-sm font-medium">Custom Frequency</Label>
            <p className="text-xs text-muted-foreground">
              Override band selection with custom frequency
            </p>
          </div>
          <Switch
            checked={useCustomFrequency}
            onCheckedChange={handleUseCustomToggle}
          />
        </div>

        {/* Custom Frequency Input */}
        {useCustomFrequency && (
          <div className="space-y-3 p-4 bg-muted/20 rounded-lg border border-dashed">
            <div className="flex items-center gap-2">
              <Target className="h-4 w-4 text-primary" />
              <Label className="text-sm font-medium">Frequency: {formatFrequency(customFrequency)}</Label>
            </div>
            
            <div className="space-y-2">
              <Slider
                value={[customFrequency]}
                onValueChange={(value) => handleCustomFrequencyChange(value[0])}
                min={0.1}
                max={150}
                step={0.1}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>0.1 GHz</span>
                <span>150 GHz</span>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-2">
              <Input
                type="number"
                value={customFrequency}
                onChange={(e) => handleCustomFrequencyChange(parseFloat(e.target.value) || 0.1)}
                min={0.1}
                max={150}
                step={0.1}
                className="text-sm"
              />
              <div className="text-xs text-muted-foreground mt-2">
                Range: 0.1 - 150 GHz
              </div>
            </div>
          </div>
        )}

        {/* Optimal Frequency Suggestion */}
        {optimalFrequency && !useCustomFrequency && (
          <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <Brain className="h-4 w-4 text-blue-600" />
              <span className="text-sm font-medium text-blue-900">Optimal Frequency</span>
            </div>
            <p className="text-sm text-blue-700">
              Based on your antenna dimensions: {formatFrequency(optimalFrequency)}
            </p>
            <Button
              variant="outline"
              size="sm"
              className="mt-2 text-blue-700 border-blue-300"
              onClick={() => handleCustomFrequencyChange(optimalFrequency)}
            >
              Use Optimal Frequency
            </Button>
          </div>
        )}

        {!useCustomFrequency && (
          <Tabs defaultValue="categories" className="w-full">
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="categories">By Category</TabsTrigger>
              <TabsTrigger value="search">Search Bands</TabsTrigger>
            </TabsList>
            
            <TabsContent value="categories" className="space-y-4">
              {/* Category Filter */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                {[
                  { id: "all", name: "All Bands", icon: "target" },
                  { id: "traditional", name: "Traditional", icon: "radio" },
                  { id: "5g6g", name: "5G/6G", icon: "zap" },
                  { id: "ism", name: "ISM", icon: "settings" },
                  { id: "wifi", name: "WiFi", icon: "wifi" },
                  { id: "medical", name: "Medical", icon: "heart" },
                  { id: "mmwave", name: "mmWave", icon: "waves" },
                  { id: "thz", name: "THz", icon: "satellite" },
                ].map(category => (
                  <Button
                    key={category.id}
                    variant={selectedCategory === category.id ? "default" : "outline"}
                    size="sm"
                    onClick={() => setSelectedCategory(category.id)}
                    className="flex items-center gap-1 text-xs"
                  >
                    {getCategoryIcon(category.id)}
                    {category.name}
                  </Button>
                ))}
              </div>

              {/* Band Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 max-h-96 overflow-y-auto">
                {filteredBands.map(band => {
                  const applications = getApplications(band)
                  const isSelected = state.selectedBand?.id === band.id
                  
                  return (
                    <div
                      key={band.id}
                      className={`p-3 rounded-lg border cursor-pointer transition-all hover:shadow-md ${
                        isSelected
                          ? 'bg-gradient-to-r from-blue-50 to-purple-50 border-blue-300 ring-2 ring-blue-200'
                          : 'bg-white hover:bg-gray-50 border-gray-200'
                      }`}
                      onClick={() => handleBandSelect(band.id)}
                    >
                      <div className="flex items-center justify-between mb-2">
                        <Badge 
                          variant="outline" 
                          style={{ borderColor: band.color, color: band.color }}
                          className="font-medium"
                        >
                          {band.name}
                        </Badge>
                        <span className="text-xs text-gray-500 font-mono">
                          {formatFrequency(band.center_freq)}
                        </span>
                      </div>
                      
                      <p className="text-xs text-gray-600 mb-2">{band.range}</p>
                      
                      <div className="flex flex-wrap gap-1">
                        {applications.map(app => (
                          <Badge key={app} variant="secondary" className="text-xs px-1 py-0">
                            {app}
                          </Badge>
                        ))}
                        {band.center_freq >= 30 && (
                          <Badge variant="secondary" className="text-xs px-1 py-0 bg-orange-100 text-orange-700">
                            mmWave
                          </Badge>
                        )}
                        {band.center_freq >= 95 && (
                          <Badge variant="secondary" className="text-xs px-1 py-0 bg-purple-100 text-purple-700">
                            THz
                          </Badge>
                        )}
                      </div>
                    </div>
                  )
                })}
              </div>
            </TabsContent>
            
            <TabsContent value="search" className="space-y-4">
              {/* Search Input */}
              <div className="relative">
                <Search className="absolute left-2 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                <Input
                  placeholder="Search frequency bands..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-8"
                />
              </div>

              {/* Search Results */}
              <div className="space-y-2 max-h-96 overflow-y-auto">
                {filteredBands.map(band => (
                  <div
                    key={band.id}
                    className={`p-3 rounded-lg border cursor-pointer transition-all hover:shadow-sm ${
                      state.selectedBand?.id === band.id
                        ? 'bg-blue-50 border-blue-300'
                        : 'bg-white hover:bg-gray-50'
                    }`}
                    onClick={() => handleBandSelect(band.id)}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Badge variant="outline" style={{ borderColor: band.color, color: band.color }}>
                          {band.name}
                        </Badge>
                        <span className="text-sm font-medium">{band.range}</span>
                      </div>
                      <span className="text-sm text-gray-500">
                        {formatFrequency(band.center_freq)}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </TabsContent>
          </Tabs>
        )}

        {/* Current Selection Info */}
        {(state.selectedBand || useCustomFrequency) && (
          <div className="p-3 bg-green-50 border border-green-200 rounded-lg">
            <div className="flex items-center gap-2 mb-1">
              <Cpu className="h-4 w-4 text-green-600" />
              <span className="text-sm font-medium text-green-900">Current Selection</span>
            </div>
            <div className="text-sm text-green-700">
              {useCustomFrequency ? (
                <p>Custom frequency: {formatFrequency(customFrequency)}</p>
              ) : state.selectedBand ? (
                <div>
                  <p>{state.selectedBand.name} - {state.selectedBand.range}</p>
                  <p>Center: {formatFrequency(state.selectedBand.center_freq)}</p>
                </div>
              ) : null}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
} 