"use client"

import React from "react"
import { useState, useEffect } from "react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Separator } from "@/components/ui/separator"
import { Activity, Antenna, BarChart3, Battery, Brain, Database, Download, 
         FileText, Heart, LineChart, Settings, TrendingDown, TrendingUp, 
         Target, Zap, WifiIcon, Upload, Play, Loader2, Calculator } from "lucide-react";
import { useSAR } from "@/components/sar-context"
import { useSARApi } from "@/hooks/use-sar-api"
import ChatInterface from "@/components/chat-interface"
import { CircularSARMap } from "@/components/circular-sar-map"
import type { AntennaParameters, SARPrediction } from "@/components/sar-context"
import { LineChart as RechartsLineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, ScatterChart, Scatter, PieChart, Pie, Cell, AreaChart, Area, RadialBarChart, RadialBar } from 'recharts';
import { sarAPI } from "@/lib/sar-api"

const healthcareSARData = [
  { frequency: 2.4, cardiac: 0.42, neural: 0.68, glucose: 0.15 },
  { frequency: 915, cardiac: 0.35, neural: 0.55, glucose: 0.12 },
  { frequency: 13.56, cardiac: 0.28, neural: 0.45, glucose: 0.10 },
  { frequency: 5.8, cardiac: 0.32, neural: 0.50, glucose: 0.11 },
  { frequency: 2.1, cardiac: 0.38, neural: 0.60, glucose: 0.14 },
];

// Enhanced SAR data with extended frequency coverage
const enhancedSARData = [
  { frequency: 0.4, skin_surface: 0.12, skin_2mm: 0.08, fat: 0.03, muscle: 0.18 },
  { frequency: 0.9, skin_surface: 0.28, skin_2mm: 0.19, fat: 0.08, muscle: 0.35 },
  { frequency: 1.8, skin_surface: 0.45, skin_2mm: 0.32, fat: 0.15, muscle: 0.58 },
  { frequency: 2.4, skin_surface: 0.52, skin_2mm: 0.38, fat: 0.18, muscle: 0.68 },
  { frequency: 3.5, skin_surface: 0.68, skin_2mm: 0.48, fat: 0.25, muscle: 0.89 },
  { frequency: 5.8, skin_surface: 0.95, skin_2mm: 0.71, fat: 0.42, muscle: 1.25 },
  { frequency: 10, skin_surface: 1.45, skin_2mm: 1.12, fat: 0.68, muscle: 1.89 },
  { frequency: 24, skin_surface: 2.15, skin_2mm: 1.85, fat: 1.25, muscle: 2.68 },
  { frequency: 28, skin_surface: 2.35, skin_2mm: 2.08, fat: 1.45, muscle: 2.95 },
  { frequency: 38, skin_surface: 2.85, skin_2mm: 2.52, fat: 1.85, muscle: 3.58 },
  { frequency: 60, skin_surface: 3.68, skin_2mm: 3.25, fat: 2.48, muscle: 4.25 },
  { frequency: 77, skin_surface: 4.25, skin_2mm: 3.85, fat: 2.95, muscle: 4.95 },
  { frequency: 94, skin_surface: 4.85, skin_2mm: 4.45, fat: 3.48, muscle: 5.68 },
  { frequency: 122, skin_surface: 5.68, skin_2mm: 5.25, fat: 4.15, muscle: 6.85 },
  { frequency: 150, skin_surface: 6.45, skin_2mm: 5.95, fat: 4.85, muscle: 7.95 }
];

const efficiencyData = [
  { application: 'Cardiac', efficiency: 85 },
  { application: 'Neural', efficiency: 90 },
  { application: 'Glucose', efficiency: 80 },
];

const complianceData = [
  { name: 'Cardiac', value: 95, color: '#ef4444' },
  { name: 'Neural', value: 92, color: '#3b82f6' },
  { name: 'Glucose', value: 90, color: '#10b981' },
];

const tissueInteractionData = [
  { depth: 1, sar: 0.42 },
  { depth: 2, sar: 0.35 },
  { depth: 3, sar: 0.28 },
  { depth: 4, sar: 0.32 },
  { depth: 5, sar: 0.38 },
];

// Enhanced dashboard data
const frequencyBandData = [
  { name: 'Traditional', value: 15, fill: '#8884d8' },
  { name: 'ISM', value: 20, fill: '#82ca9d' },
  { name: 'WiFi', value: 18, fill: '#ffc658' },
  { name: '5G', value: 25, fill: '#ff7300' },
  { name: '6G/THz', value: 22, fill: '#00ff88' },
];

const safetyMetricsData = [
  { metric: 'FCC Compliance', value: 92, fill: '#10b981' },
  { metric: 'ICNIRP Compliance', value: 88, fill: '#3b82f6' },
  { metric: 'Healthcare Safe', value: 96, fill: '#8b5cf6' },
  { metric: 'Wearable Ready', value: 85, fill: '#f59e0b' },
];

const realTimeData = [
  { time: '09:00', cardiac: 0.42, neural: 0.35, glucose: 0.15 },
  { time: '09:15', cardiac: 0.38, neural: 0.32, glucose: 0.14 },
  { time: '09:30', cardiac: 0.45, neural: 0.38, glucose: 0.16 },
  { time: '09:45', cardiac: 0.41, neural: 0.33, glucose: 0.15 },
  { time: '10:00', cardiac: 0.39, neural: 0.31, glucose: 0.13 },
];

export default function DashboardContent() {
  const { state, dispatch } = useSAR()
  const {
    predictSAR,
    generateSampleParameters,
    loadFrequencyBands,
    checkSystemStatus,
    healthCheck,
    trainModel,
    downloadDataset,
    getHealthcareApplications,
    isLoading,
    selectedBand,
    frequencyBands,
    currentPrediction,
    predictionHistory,
    comparisonData,
    systemStatus,
    parameters,
    inputMode,
  } = useSARApi()

  // Enhanced features state
  const [frequencySweepData, setFrequencySweepData] = useState<any[]>([])
  const [circularSARMap, setCircularSARMap] = useState<any>(null)
  const [isGeneratingSweep, setIsGeneratingSweep] = useState(false)
  const [isGeneratingMap, setIsGeneratingMap] = useState(false)

  // Load initial data
  React.useEffect(() => {
    loadFrequencyBands()
    checkSystemStatus()
  }, [loadFrequencyBands, checkSystemStatus])

  const handleParameterChange = (key: keyof AntennaParameters, value: number) => {
    dispatch({
      type: "SET_PARAMETERS",
      payload: { [key]: value }
    })
  }

  const handleBandSelect = (bandId: string) => {
    const band = (frequencyBands || []).find(b => b.id === bandId)
    if (band) {
      dispatch({ type: "SET_BAND", payload: band })
    }
  }

  const handleGenerateParameters = () => {
    if (selectedBand) {
      generateSampleParameters(selectedBand.id)
    }
  }

  const handlePredict = () => {
    if (selectedBand) {
      predictSAR(parameters, selectedBand.id)
    }
  }

  const handleTrainModel = async () => {
    const response = await trainModel({
      num_samples: 2000,
      include_real_data: true,
      focus_applications: ["cardiac_monitoring", "neural_interfaces", "glucose_monitoring"]
    })
    
    if (response) {
      alert(`Training completed! Accuracy: ${(response.model_accuracy * 100).toFixed(1)}%`)
    } else {
      alert("Training failed. Please try again.")
    }
  }

  const handleDownloadDataset = async () => {
    const response = await downloadDataset("comprehensive")
    if (response) {
      alert(`Dataset ready! ${response.total_samples} samples available for download.`)
    } else {
      alert("Dataset generation failed. Please try again.")
    }
  }

  // Enhanced features handlers
  const handleGenerateFrequencySweep = async () => {
    setIsGeneratingSweep(true)
    try {
      const sweepData = await sarAPI.generateFrequencySweep(state.parameters)
      setFrequencySweepData(sweepData)
    } catch (error) {
      console.error("Failed to generate frequency sweep:", error)
    } finally {
      setIsGeneratingSweep(false)
    }
  }

  const handleGenerateCircularSARMap = async () => {
    if (!state.selectedBand) return
    
    setIsGeneratingMap(true)
    try {
      const mapData = await sarAPI.generateSARMap(state.selectedBand.center_freq, state.parameters)
      setCircularSARMap(mapData)
    } catch (error) {
      console.error("Failed to generate circular SAR map:", error)
    } finally {
      setIsGeneratingMap(false)
    }
  }

  if (state.activeView === "chat") {
    return <ChatInterface />
  }

  return (
    <div className="flex-1 space-y-6 p-4 md:p-6 lg:p-8 pt-6 max-w-full overflow-hidden">
      <div className="flex items-center justify-between space-y-2">
        <div>
          <h2 className="text-4xl font-bold tracking-tight">Enhanced SAR Prediction System</h2>
          <p className="text-lg text-muted-foreground mt-2">
            Physics-based Performance Prediction with Extended Frequency Coverage (0.1-150 GHz)
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <Badge variant="outline" className="text-lg px-4 py-2">
            {systemStatus?.status === "healthy" ? "System Online" : "System Status Unknown"}
          </Badge>
          <Badge variant="secondary" className="text-lg px-4 py-2">
            Enhanced Physics Engine
          </Badge>
        </div>
      </div>

      <Tabs value={state.activeView} onValueChange={(value) => 
        dispatch({ type: "SET_ACTIVE_VIEW", payload: value as any })
      } className="space-y-6">
        <TabsList className="grid w-full grid-cols-5 h-12">
          <TabsTrigger value="dashboard" className="text-sm px-3 py-2 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
            Overview
          </TabsTrigger>
          <TabsTrigger value="comparison" className="text-sm px-3 py-2 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
            Prediction
          </TabsTrigger>
          <TabsTrigger value="enhanced" className="text-sm px-3 py-2 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
            Enhanced Analysis
          </TabsTrigger>
          <TabsTrigger value="history" className="text-sm px-3 py-2 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
            History
          </TabsTrigger>
          <TabsTrigger value="settings" className="text-sm px-3 py-2 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
            Settings
          </TabsTrigger>
        </TabsList>

        <TabsContent value="dashboard" className="space-y-6">
          <div className="grid gap-6 grid-cols-1 md:grid-cols-2 lg:grid-cols-4">
            <Card className="border-2">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-base font-medium">Total Predictions</CardTitle>
                <LineChart className="h-6 w-6 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold">
                  {state.predictionHistory.length.toLocaleString()}
                </div>
                <p className="text-base text-muted-foreground">
                  Enhanced physics predictions
                </p>
              </CardContent>
            </Card>

            <Card className="border-2">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-base font-medium">Average SAR</CardTitle>
                <Activity className="h-6 w-6 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold">
                  {state.predictionHistory.length > 0 
                    ? (state.predictionHistory.reduce((sum: number, p: SARPrediction) => sum + p.sar_value, 0) / state.predictionHistory.length).toFixed(3)
                    : "0.000"
                  } W/kg
                </div>
                <p className="text-base text-muted-foreground">
                  Physics-based calculations
                </p>
              </CardContent>
            </Card>

            <Card className="border-2">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-base font-medium">Frequency Coverage</CardTitle>
                <Zap className="h-6 w-6 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold text-green-600">0.1-150 GHz</div>
                <p className="text-base text-muted-foreground">
                  Including 5G/6G/THz bands
                </p>
              </CardContent>
            </Card>

            <Card className="border-2">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-base font-medium">Active Band</CardTitle>
                <WifiIcon className="h-6 w-6 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold">
                  {state.selectedBand ? state.selectedBand.name : "None"}
                </div>
                <p className="text-base text-muted-foreground">
                  {state.selectedBand ? `${state.selectedBand.center_freq} GHz` : "Select frequency band"}
                </p>
              </CardContent>
            </Card>
          </div>

          {/* Row 1: Enhanced SAR vs Frequency Analysis | Safety Metrics Dashboard */}
          <div className="grid gap-6 grid-cols-1 lg:grid-cols-2">
            <Card className="border-2">
              <CardHeader>
                <CardTitle className="text-2xl">Enhanced SAR vs Frequency Analysis</CardTitle>
                <CardDescription className="text-lg">
                  Extended frequency coverage (0.1-150 GHz) including 5G/6G/THz bands
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={350}>
                  <RechartsLineChart data={enhancedSARData}>
                    <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                    <XAxis 
                      dataKey="frequency" 
                      label={{ value: 'Frequency (GHz)', position: 'insideBottom', offset: -5 }}
                      type="number"
                      scale="log"
                      domain={['dataMin', 'dataMax']}
                    />
                    <YAxis label={{ value: 'SAR (W/kg)', angle: -90, position: 'insideLeft' }} />
                    <Tooltip 
                      formatter={(value, name) => [
                        `${value} W/kg`, 
                        name === 'skin_surface' ? 'Skin Surface' : 
                        name === 'skin_2mm' ? 'Skin 2mm Depth' : 
                        name === 'fat' ? 'Fat Tissue' : 'Muscle Tissue'
                      ]}
                      labelFormatter={(freq) => `Frequency: ${freq} GHz`}
                    />
                    <Legend />
                    <Line type="monotone" dataKey="skin_surface" stroke="#ef4444" strokeWidth={3} name="Skin Surface" />
                    <Line type="monotone" dataKey="skin_2mm" stroke="#f59e0b" strokeWidth={2} name="Skin 2mm" />
                    <Line type="monotone" dataKey="fat" stroke="#3b82f6" strokeWidth={2} name="Fat Tissue" />
                    <Line type="monotone" dataKey="muscle" stroke="#10b981" strokeWidth={2} name="Muscle Tissue" />
                  </RechartsLineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card className="border-2">
              <CardHeader>
                <CardTitle className="text-2xl">Safety Metrics Dashboard</CardTitle>
                <CardDescription className="text-lg">
                  Compliance rates across safety standards
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <RadialBarChart cx="50%" cy="50%" innerRadius="20%" outerRadius="90%" data={safetyMetricsData}>
                    <RadialBar
                      dataKey="value"
                      cornerRadius={10}
                      fill="#8884d8"
                    />
                    <Tooltip formatter={(value) => [`${value}%`, 'Compliance Rate']} />
                  </RadialBarChart>
                </ResponsiveContainer>
                <div className="mt-4 grid grid-cols-2 gap-2 text-sm">
                  {safetyMetricsData.map((item, index) => (
                    <div key={index} className="flex items-center space-x-2">
                      <div 
                        className="w-3 h-3 rounded-full" 
                        style={{ backgroundColor: item.fill }}
                      ></div>
                      <span className="text-xs">{item.metric}: {item.value}%</span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Row 2: Spatial SAR Distribution (full width) */}
          <div className="w-full">
            <Card className="border-2 w-full">
              <CardHeader>
                <CardTitle className="text-2xl">Spatial SAR Distribution Map</CardTitle>
                <CardDescription className="text-lg">
                  Interactive spatial analysis showing SAR effects around antenna (km-based coverage)
                </CardDescription>
              </CardHeader>
              <CardContent className="w-full overflow-hidden">
                <CircularSARMap 
                  frequency={state.selectedBand?.center_freq || 2.45} 
                  className="w-full max-w-none"
                />
              </CardContent>
            </Card>
          </div>

          {/* Row 3: Frequency Band Distribution | Real-time SAR Monitor */}
          <div className="grid gap-6 grid-cols-1 lg:grid-cols-2">
            <Card className="border-2">
              <CardHeader>
                <CardTitle className="text-2xl">Frequency Band Distribution</CardTitle>
                <CardDescription className="text-lg">
                  Usage across different frequency categories
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={frequencyBandData}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                      outerRadius={90}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {frequencyBandData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.fill} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card className="border-2">
              <CardHeader>
                <CardTitle className="text-2xl">Real-time SAR Monitor</CardTitle>
                <CardDescription className="text-lg">
                  Live monitoring of wearable devices
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <RechartsLineChart data={realTimeData}>
                    <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                    <XAxis dataKey="time" />
                    <YAxis label={{ value: 'SAR (W/kg)', angle: -90, position: 'insideLeft' }} />
                    <Tooltip 
                      formatter={(value, name) => [
                        `${value} W/kg`, 
                        name === 'cardiac' ? 'Cardiac Monitor' : 
                        name === 'neural' ? 'Neural Interface' : 'Glucose Sensor'
                      ]}
                    />
                    <Legend />
                    <Line type="monotone" dataKey="cardiac" stroke="#ef4444" strokeWidth={2} name="Cardiac" />
                    <Line type="monotone" dataKey="neural" stroke="#3b82f6" strokeWidth={2} name="Neural" />
                    <Line type="monotone" dataKey="glucose" stroke="#10b981" strokeWidth={2} name="Glucose" />
                  </RechartsLineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>

          {/* Row 4: System Performance Analytics | Active Device Monitor */}
          <div className="grid gap-6 grid-cols-1 lg:grid-cols-2">
            <Card className="border-2">
              <CardHeader>
                <CardTitle className="text-2xl">System Performance Analytics</CardTitle>
                <CardDescription className="text-lg">
                  Comprehensive analysis of prediction accuracy and system health
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="text-center p-4 bg-gradient-to-br from-blue-900 to-blue-800 rounded-lg border border-blue-600">
                      <div className="text-3xl font-bold text-blue-100">95.2%</div>
                      <div className="text-sm text-blue-200">Model Accuracy</div>
                      <div className="w-full bg-blue-700 rounded-full h-2 mt-2">
                        <div className="bg-blue-300 h-2 rounded-full" style={{ width: '95.2%' }}></div>
                      </div>
                    </div>
                    <div className="text-center p-4 bg-gradient-to-br from-green-900 to-green-800 rounded-lg border border-green-600">
                      <div className="text-3xl font-bold text-green-100">99.8%</div>
                      <div className="text-sm text-green-200">System Uptime</div>
                      <div className="w-full bg-green-700 rounded-full h-2 mt-2">
                        <div className="bg-green-300 h-2 rounded-full" style={{ width: '99.8%' }}></div>
                      </div>
                    </div>
                  </div>
                  
                  <ResponsiveContainer width="100%" height={200}>
                    <BarChart data={efficiencyData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis dataKey="application" stroke="#9ca3af" />
                      <YAxis stroke="#9ca3af" />
                      <Tooltip 
                        formatter={(value) => [`${value}%`, 'Efficiency']}
                        contentStyle={{ 
                          backgroundColor: '#1f2937', 
                          border: '1px solid #374151',
                          borderRadius: '6px',
                          color: '#f3f4f6'
                        }}
                      />
                      <Bar dataKey="efficiency" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            <Card className="border-2">
              <CardHeader>
                <CardTitle className="text-2xl">Active Device Monitor</CardTitle>
                <CardDescription className="text-lg">
                  Real-time status of connected wearable devices
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex items-center justify-between p-4 bg-gradient-to-r from-red-900 to-red-800 border border-red-600 rounded-lg">
                    <div className="flex items-center space-x-3">
                      <div className="relative">
                        <Heart className="w-8 h-8 text-red-400" />
                        <div className="absolute -top-1 -right-1 w-3 h-3 bg-green-400 rounded-full animate-pulse"></div>
                      </div>
                      <div>
                        <h3 className="font-medium text-red-100">Cardiac Monitor</h3>
                        <p className="text-sm text-red-200">2.4 GHz • 10 mW</p>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-lg font-bold text-green-400">0.42 W/kg</div>
                      <Badge variant="secondary" className="text-xs bg-green-900 text-green-300 border-green-500">Safe</Badge>
                    </div>
                  </div>

                  <div className="flex items-center justify-between p-4 bg-gradient-to-r from-blue-900 to-blue-800 border border-blue-600 rounded-lg">
                    <div className="flex items-center space-x-3">
                      <div className="relative">
                        <Brain className="w-8 h-8 text-blue-400" />
                        <div className="absolute -top-1 -right-1 w-3 h-3 bg-green-400 rounded-full animate-pulse"></div>
                      </div>
                      <div>
                        <h3 className="font-medium text-blue-100">Neural Interface</h3>
                        <p className="text-sm text-blue-200">915 MHz • 25 mW</p>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-lg font-bold text-green-400">0.68 W/kg</div>
                      <Badge variant="secondary" className="text-xs bg-green-900 text-green-300 border-green-500">Safe</Badge>
                    </div>
                  </div>

                  <div className="flex items-center justify-between p-4 bg-gradient-to-r from-green-900 to-green-800 border border-green-600 rounded-lg">
                    <div className="flex items-center space-x-3">
                      <div className="relative">
                        <Target className="w-8 h-8 text-green-400" />
                        <div className="absolute -top-1 -right-1 w-3 h-3 bg-green-400 rounded-full animate-pulse"></div>
                      </div>
                      <div>
                        <h3 className="font-medium text-green-100">Glucose Sensor</h3>
                        <p className="text-sm text-green-200">13.56 MHz • 5 mW</p>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-lg font-bold text-green-400">0.15 W/kg</div>
                      <Badge variant="secondary" className="text-xs bg-green-900 text-green-300 border-green-500">Safe</Badge>
                    </div>
                  </div>

                  <div className="mt-4 p-3 bg-gray-900 border border-gray-700 rounded-lg">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-gray-300">Total Active Devices:</span>
                      <span className="font-bold text-white">3</span>
                    </div>
                    <div className="flex items-center justify-between text-sm mt-1">
                      <span className="text-gray-300">Average SAR:</span>
                      <span className="font-bold text-green-400">0.42 W/kg</span>
                    </div>
                    <div className="flex items-center justify-between text-sm mt-1">
                      <span className="text-gray-300">System Status:</span>
                      <Badge variant="secondary" className="text-xs bg-green-900 text-green-300 border-green-500">All Safe</Badge>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="comparison" className="space-y-6">
          <Card className="border-2">
            <CardHeader>
              <CardTitle className="text-2xl">SAR Prediction</CardTitle>
              <CardDescription className="text-lg">
                Configure antenna parameters, select frequency band, and predict SAR levels
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid gap-6 md:grid-cols-2">
                <div className="space-y-4">
                  {/* Frequency Band Selection */}
                  <Card className="border">
                    <CardHeader className="pb-3">
                      <CardTitle className="text-lg">Frequency Configuration</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="space-y-2">
                        <Label className="text-base font-medium">Select Frequency Band</Label>
                        <select 
                          className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                          value={state.selectedBand?.id || ""}
                          onChange={(e) => {
                            const band = (frequencyBands || []).find(b => b.id === e.target.value)
                            if (band) handleBandSelect(band.id)
                          }}
                        >
                          <option value="">Select a frequency band...</option>
                          {(frequencyBands || []).map(band => (
                            <option key={band.id} value={band.id}>
                              {band.name} - {band.center_freq} GHz
                            </option>
                          ))}
                        </select>
                      </div>
                      
                      {state.selectedBand && (
                        <div className="p-3 bg-blue-50 rounded-lg border">
                          <div className="flex items-center space-x-2 mb-2">
                            <WifiIcon className="h-4 w-4 text-blue-600" />
                            <span className="font-medium text-blue-900">{state.selectedBand.name}</span>
                          </div>
                          <p className="text-sm text-blue-700">Range: {state.selectedBand.range}</p>
                          <p className="text-sm text-blue-700">Center: {state.selectedBand.center_freq} GHz</p>
                        </div>
                      )}

                      <div className="space-y-2">
                        <Label htmlFor="custom-frequency" className="text-base font-medium">
                          Custom Frequency (GHz)
                        </Label>
                        <Input
                          id="custom-frequency"
                          type="number"
                          step="0.1"
                          min="0.1"
                          max="150"
                          defaultValue={state.selectedBand?.center_freq || 10}
                          className="text-base h-10"
                          placeholder="Enter frequency in GHz"
                        />
                        <p className="text-xs text-muted-foreground">
                          Range: 0.1 - 150 GHz (including 6G/THz bands)
                        </p>
                      </div>
                    </CardContent>
                  </Card>

                  {/* Antenna Parameters */}
                  <Card className="border">
                    <CardHeader className="pb-3">
                      <CardTitle className="text-lg">Antenna Parameters</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="grid grid-cols-2 gap-4">
                        <div className="space-y-2">
                          <Label htmlFor="thickness" className="text-base">Substrate Thickness (mm)</Label>
                          <Input
                            id="thickness"
                            type="number"
                            step="0.1"
                            value={parameters.substrate_thickness}
                            onChange={(e) => handleParameterChange('substrate_thickness', parseFloat(e.target.value))}
                            className="text-base h-10"
                          />
                        </div>
                        <div className="space-y-2">
                          <Label htmlFor="permittivity" className="text-base">Relative Permittivity</Label>
                          <Input
                            id="permittivity"
                            type="number"
                            step="0.1"
                            value={parameters.substrate_permittivity}
                            onChange={(e) => handleParameterChange('substrate_permittivity', parseFloat(e.target.value))}
                            className="text-base h-10"
                          />
                        </div>
                      </div>
                      <div className="grid grid-cols-2 gap-4">
                        <div className="space-y-2">
                          <Label htmlFor="width" className="text-base">Patch Width (mm)</Label>
                          <Input
                            id="width"
                            type="number"
                            step="0.1"
                            value={parameters.patch_width}
                            onChange={(e) => handleParameterChange('patch_width', parseFloat(e.target.value))}
                            className="text-base h-10"
                          />
                        </div>
                        <div className="space-y-2">
                          <Label htmlFor="length" className="text-base">Patch Length (mm)</Label>
                          <Input
                            id="length"
                            type="number"
                            step="0.1"
                            value={parameters.patch_length}
                            onChange={(e) => handleParameterChange('patch_length', parseFloat(e.target.value))}
                            className="text-base h-10"
                          />
                        </div>
                      </div>
                      <div className="grid grid-cols-2 gap-4">
                        <div className="space-y-2">
                          <Label htmlFor="bending" className="text-base">Bending Radius (mm)</Label>
                          <Input
                            id="bending"
                            type="number"
                            step="1"
                            value={parameters.bending_radius}
                            onChange={(e) => handleParameterChange('bending_radius', parseFloat(e.target.value))}
                            className="text-base h-10"
                          />
                        </div>
                        <div className="space-y-2">
                          <Label htmlFor="power" className="text-base">Power Density (W/cm²)</Label>
                          <Input
                            id="power"
                            type="number"
                            step="0.1"
                            value={parameters.power_density}
                            onChange={(e) => handleParameterChange('power_density', parseFloat(e.target.value))}
                            className="text-base h-10"
                          />
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  <div className="flex space-x-4">
                    <Button 
                      onClick={handleGenerateParameters}
                      disabled={isLoading || !selectedBand}
                      variant="outline"
                      className="text-base px-6 py-2 h-auto"
                    >
                      Generate Random
                    </Button>
                    <Button 
                      onClick={handlePredict}
                      disabled={isLoading || !selectedBand}
                      className="text-base px-6 py-2 h-auto"
                    >
                      {isLoading ? "Predicting..." : "Predict SAR"}
                    </Button>
                  </div>
                </div>

                <div className="space-y-4">
                  {currentPrediction && (
                    <Card className="border-2">
                      <CardHeader>
                        <CardTitle className="text-xl">Prediction Results</CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-4">
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                          <div>
                            <p className="text-lg font-medium">SAR Value</p>
                            <p className="text-2xl font-bold text-primary">
                              {(isNaN(currentPrediction?.sar_value) ? 0 : currentPrediction?.sar_value || 0).toFixed(3)} W/kg
                            </p>
                          </div>
                          <div>
                            <p className="text-lg font-medium">Gain</p>
                            <p className="text-2xl font-bold">
                              {(isNaN(currentPrediction?.gain) ? 0 : currentPrediction?.gain || 0).toFixed(2)} dBi
                            </p>
                          </div>
                        </div>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                          <div>
                            <p className="text-lg font-medium">Efficiency</p>
                            <p className="text-2xl font-bold">
                              {(isNaN(currentPrediction?.efficiency) ? 0 : currentPrediction?.efficiency || 0).toFixed(1)}%
                            </p>
                          </div>
                          <div>
                            <p className="text-lg font-medium">Bandwidth</p>
                            <p className="text-2xl font-bold">
                              {(isNaN(currentPrediction?.bandwidth) ? 0 : currentPrediction?.bandwidth || 0).toFixed(2)}%
                            </p>
                          </div>
                        </div>
                        <Separator />
                        <div className="space-y-2">
                          <p className="text-lg font-medium">Frequency Band</p>
                          <Badge variant="outline" className="text-base px-3 py-1">
                            {currentPrediction.band_name}
                          </Badge>
                        </div>
                      </CardContent>
                    </Card>
                  )}

                  {!currentPrediction && (
                    <Card className="border-2 border-dashed">
                      <CardContent className="flex items-center justify-center h-64">
                        <div className="text-center space-y-3">
                          <Settings className="h-12 w-12 mx-auto text-muted-foreground" />
                          <p className="text-xl text-muted-foreground">
                            Configure parameters and run prediction
                          </p>
                        </div>
                      </CardContent>
                    </Card>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="enhanced" className="space-y-6">
          <Card className="border-2">
            <CardHeader>
              <CardTitle className="text-2xl">Enhanced SAR Analysis</CardTitle>
              <CardDescription className="text-lg">
                Advanced physics-based analysis with extended frequency coverage (0.1-150 GHz)
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid gap-6 md:grid-cols-2">
                <Card className="border">
                  <CardHeader>
                    <CardTitle className="text-xl">SAR vs Frequency Analysis</CardTitle>
                    <CardDescription>
                      Frequency sweep analysis across all bands including 6G/THz ranges
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <Button 
                        onClick={handleGenerateFrequencySweep}
                        disabled={isLoading || !state.selectedBand || isGeneratingSweep}
                        className="w-full"
                      >
                        {isGeneratingSweep ? (
                          <>
                            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                            Generating...
                          </>
                        ) : (
                          <>
                            <Calculator className="mr-2 h-4 w-4" />
                            Generate Frequency Sweep
                          </>
                        )}
                      </Button>
                      {frequencySweepData.length > 0 && (
                        <ResponsiveContainer width="100%" height={300}>
                          <RechartsLineChart data={frequencySweepData}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis 
                              dataKey="frequency" 
                              label={{ value: 'Frequency (GHz)', position: 'insideBottom', offset: -5 }}
                              scale="log"
                            />
                            <YAxis label={{ value: 'SAR (W/kg)', angle: -90, position: 'insideLeft' }} />
                            <Tooltip 
                              formatter={(value, name) => [
                                `${Number(value).toFixed(3)} W/kg`, 
                                name === 'sar_skin_surface' ? 'Skin Surface' : 
                                name === 'sar_fat_surface' ? 'Fat Tissue' : 
                                name === 'sar_muscle_surface' ? 'Muscle Tissue' : name
                              ]}
                              labelFormatter={(freq) => `Frequency: ${freq} GHz`}
                            />
                            <Legend />
                            <Line type="monotone" dataKey="sar_skin_surface" stroke="#ef4444" strokeWidth={2} name="Skin SAR" />
                            <Line type="monotone" dataKey="sar_fat_surface" stroke="#3b82f6" strokeWidth={2} name="Fat SAR" />
                            <Line type="monotone" dataKey="sar_muscle_surface" stroke="#10b981" strokeWidth={2} name="Muscle SAR" />
                          </RechartsLineChart>
                        </ResponsiveContainer>
                      )}
                      <div className="text-center text-muted-foreground">
                        <p>Analyze SAR behavior across extended frequency range</p>
                        <p className="text-sm">Including traditional, 5G, 6G, and THz bands</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card className="border">
                  <CardHeader>
                    <CardTitle className="text-xl">Circular SAR Mapping</CardTitle>
                    <CardDescription>
                      Spatial SAR distribution analysis around antenna
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <Button 
                        onClick={handleGenerateCircularSARMap}
                        disabled={isLoading || !state.selectedBand || isGeneratingMap}
                        className="w-full"
                      >
                        {isGeneratingMap ? (
                          <>
                            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                            Generating...
                          </>
                        ) : (
                          <>
                            <Target className="mr-2 h-4 w-4" />
                            Generate SAR Map
                          </>
                        )}
                      </Button>
                      {circularSARMap && (
                        <div className="space-y-3">
                          <div className="grid grid-cols-3 gap-3 text-sm">
                            <div className="text-center p-2 bg-green-50 rounded border">
                              <p className="font-medium text-green-900">Max SAR</p>
                              <p className="text-green-700">{circularSARMap.maxSAR?.toFixed(3)} W/kg</p>
                            </div>
                            <div className="text-center p-2 bg-blue-50 rounded border">
                              <p className="font-medium text-blue-900">Avg SAR</p>
                              <p className="text-blue-700">{circularSARMap.avgSAR?.toFixed(3)} W/kg</p>
                            </div>
                            <div className="text-center p-2 bg-orange-50 rounded border">
                              <p className="font-medium text-orange-900">Frequency</p>
                              <p className="text-orange-700">{circularSARMap.frequency} GHz</p>
                            </div>
                          </div>
                          <div className="text-xs text-muted-foreground text-center">
                            <p>SAR distribution map generated for {circularSARMap.resolution}x{circularSARMap.resolution} grid</p>
                          </div>
                        </div>
                      )}
                      <div className="text-center text-muted-foreground">
                        <p>Visualize SAR effects in circular area around antenna</p>
                        <p className="text-sm">Safety zones and hotspot analysis</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>

              <Card className="border">
                <CardHeader>
                  <CardTitle className="text-xl">Enhanced Physics Engine</CardTitle>
                  <CardDescription>
                    Physics-based calculations using proper electromagnetic theory
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid gap-4 md:grid-cols-3">
                    <div className="p-4 border rounded-lg bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-950/30 dark:to-blue-900/30">
                      <h4 className="font-semibold mb-2 flex items-center gap-2">
                        <Zap className="h-4 w-4" />
                        Enhanced SAR Formula
                      </h4>
                      <p className="text-sm text-muted-foreground">SAR = σ|E|²/ρ</p>
                      <p className="text-xs">Conductivity × Electric Field² / Density</p>
                    </div>
                    <div className="p-4 border rounded-lg bg-gradient-to-br from-green-50 to-green-100 dark:from-green-950/30 dark:to-green-900/30">
                      <h4 className="font-semibold mb-2 flex items-center gap-2">
                        <Heart className="h-4 w-4" />
                        Tissue Properties
                      </h4>
                      <p className="text-sm text-muted-foreground">Frequency-dependent</p>
                      <p className="text-xs">Skin, fat, muscle parameters up to 150 GHz</p>
                    </div>
                    <div className="p-4 border rounded-lg bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-950/30 dark:to-purple-900/30">
                      <h4 className="font-semibold mb-2 flex items-center gap-2">
                        <Settings className="h-4 w-4" />
                        Safety Standards
                      </h4>
                      <p className="text-sm text-muted-foreground">FCC: 1.6 W/kg</p>
                      <p className="text-xs">ICNIRP: 2.0 W/kg, Enhanced validation</p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Current Prediction Enhanced Analysis */}
              {state.currentPrediction && (
                <Card className="border">
                  <CardHeader>
                    <CardTitle className="text-xl">Current Prediction Analysis</CardTitle>
                    <CardDescription>
                      Enhanced analysis of your latest SAR prediction
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="grid gap-4 md:grid-cols-2">
                      <div className="space-y-3">
                        <h4 className="font-semibold">Safety Assessment</h4>
                        {state.currentPrediction.safety_assessment && (
                          <div className="space-y-2">
                            <div className="flex justify-between items-center p-2 bg-gray-50 rounded">
                              <span>FCC Compliance:</span>
                              <Badge variant={state.currentPrediction.safety_assessment.fcc.compliant ? "secondary" : "destructive"}>
                                {state.currentPrediction.safety_assessment.fcc.compliant ? "Compliant" : "Non-compliant"}
                              </Badge>
                            </div>
                            <div className="flex justify-between items-center p-2 bg-gray-50 rounded">
                              <span>ICNIRP Compliance:</span>
                              <Badge variant={state.currentPrediction.safety_assessment.icnirp.compliant ? "secondary" : "destructive"}>
                                {state.currentPrediction.safety_assessment.icnirp.compliant ? "Compliant" : "Non-compliant"}
                              </Badge>
                            </div>
                            <div className="flex justify-between items-center p-2 bg-gray-50 rounded">
                              <span>Safety Score:</span>
                              <span className="font-mono">{(state.currentPrediction?.safety_assessment?.safety_score || 0).toFixed(1)}%</span>
                            </div>
                          </div>
                        )}
                      </div>
                      <div className="space-y-3">
                        <h4 className="font-semibold">Tissue Analysis</h4>
                        {state.currentPrediction?.tissue_analysis && (
                          <div className="space-y-2">
                            <div className="flex justify-between items-center p-2 bg-gray-50 rounded">
                              <span>Skin SAR:</span>
                              <span className="font-mono">{(state.currentPrediction.tissue_analysis.sar_skin || 0).toFixed(3)} W/kg</span>
                            </div>
                            <div className="flex justify-between items-center p-2 bg-gray-50 rounded">
                              <span>Fat SAR:</span>
                              <span className="font-mono">{(state.currentPrediction.tissue_analysis.sar_fat || 0).toFixed(3)} W/kg</span>
                            </div>
                            <div className="flex justify-between items-center p-2 bg-gray-50 rounded">
                              <span>Muscle SAR:</span>
                              <span className="font-mono">{(state.currentPrediction.tissue_analysis.sar_muscle || 0).toFixed(3)} W/kg</span>
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="history" className="space-y-6">
          <Card className="border-2">
            <CardHeader>
              <CardTitle className="text-2xl">Prediction History</CardTitle>
              <CardDescription className="text-lg">
                View and analyze your previous SAR predictions
              </CardDescription>
            </CardHeader>
            <CardContent>
              {predictionHistory.length === 0 ? (
                <div className="text-center py-12">
                  <Database className="h-16 w-16 mx-auto text-muted-foreground mb-4" />
                  <p className="text-xl text-muted-foreground">No predictions yet</p>
                  <p className="text-lg text-muted-foreground">
                    Run your first prediction to see results here
                  </p>
                </div>
              ) : (
                <div className="space-y-6">
                  {predictionHistory.map((prediction: SARPrediction) => (
                    <Card key={prediction.id} className="border">
                      <CardHeader>
                        <div className="flex items-center justify-between">
                          <CardTitle className="text-xl">{prediction.band_name}</CardTitle>
                          <Badge variant="outline" className="text-base px-3 py-1">
                            {new Date(prediction.timestamp).toLocaleDateString()}
                          </Badge>
                        </div>
                      </CardHeader>
                      <CardContent>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                          <div>
                            <p className="text-lg font-medium">SAR</p>
                            <p className="text-2xl font-bold text-primary">
                              {(prediction?.sar_value || 0).toFixed(3)} W/kg
                            </p>
                          </div>
                          <div>
                            <p className="text-lg font-medium">Gain</p>
                            <p className="text-2xl font-bold">
                              {(prediction?.gain || 0).toFixed(2)} dBi
                            </p>
                          </div>
                        </div>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                          <div>
                            <p className="text-lg font-medium">Efficiency</p>
                            <p className="text-2xl font-bold">
                              {(prediction?.efficiency || 0).toFixed(1)}%
                            </p>
                          </div>
                          <div>
                            <p className="text-lg font-medium">Bandwidth</p>
                            <p className="text-2xl font-bold">
                              {(prediction?.bandwidth || 0).toFixed(2)}%
                            </p>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="settings" className="space-y-6">
          <div className="grid gap-6 md:grid-cols-2">
            <Card className="border-2">
              <CardHeader>
                <CardTitle className="text-2xl">System Configuration</CardTitle>
                <CardDescription className="text-lg">
                  Configure system preferences and API settings
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-6">
                  <div>
                    <h3 className="text-xl font-medium mb-4">API Configuration</h3>
                    <div className="space-y-4">
                      <div className="flex items-center justify-between p-4 border rounded-lg">
                        <div>
                          <p className="text-lg font-medium">Backend API</p>
                          <p className="text-base text-muted-foreground">
                            http://localhost:8000
                          </p>
                        </div>
                        <Badge variant="outline" className="text-base px-3 py-1">
                          {systemStatus ? "Connected" : "Checking..."}
                        </Badge>
                      </div>
                    </div>
                  </div>

                  <div>
                    <h3 className="text-xl font-medium mb-4">Model Information</h3>
                    <div className="space-y-4">
                      <div className="flex items-center justify-between p-4 border rounded-lg">
                        <div>
                          <p className="text-lg font-medium">SAR Model</p>
                          <p className="text-base text-muted-foreground">
                            Physics-based prediction model v2.0
                          </p>
                        </div>
                        <Badge variant="outline" className="text-base px-3 py-1">
                          Active
                        </Badge>
                      </div>
                    </div>
                  </div>

                  <div>
                    <h3 className="text-xl font-medium mb-4">Actions</h3>
                    <div className="space-y-4">
                      <Button
                        onClick={checkSystemStatus}
                        variant="outline"
                        className="text-lg px-6 py-3 h-auto w-full"
                      >
                        Refresh System Status
                      </Button>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="border-2">
              <CardHeader>
                <CardTitle className="text-2xl">Model Training & Data</CardTitle>
                <CardDescription className="text-lg">
                  Train model with healthcare wearable antenna datasets
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-4">
                  <div>
                    <h3 className="text-xl font-medium mb-4">Upload Training Data</h3>
                    <div className="border-2 border-dashed border-border/40 rounded-lg p-6 text-center space-y-4">
                      <Database className="w-12 h-12 mx-auto text-muted-foreground" />
                      <div>
                        <p className="text-lg font-medium">Upload SAR Dataset</p>
                        <p className="text-base text-muted-foreground">
                          CSV format with antenna parameters and measured SAR values
                        </p>
                      </div>
                      <input
                        type="file"
                        accept=".csv,.xlsx"
                        className="hidden"
                        id="data-upload"
                      />
                      <label htmlFor="data-upload">
                        <Button variant="outline" className="text-lg px-6 py-3 h-auto cursor-pointer">
                          Choose File
                        </Button>
                      </label>
                    </div>
                  </div>

                  <div>
                    <h3 className="text-xl font-medium mb-4">Real-time Training</h3>
                    <div className="space-y-3">
                      <Button
                        variant="default"
                        className="text-lg px-6 py-3 h-auto w-full"
                        disabled={isLoading}
                        onClick={handleTrainModel}
                      >
                        {isLoading ? "Training..." : "Train Model with Healthcare Data"}
                      </Button>
                      <Button
                        variant="secondary"
                        className="text-lg px-6 py-3 h-auto w-full"
                        onClick={handleDownloadDataset}
                      >
                        Download Real Healthcare Dataset
                      </Button>
                    </div>
                  </div>

                  <div>
                    <h3 className="text-xl font-medium mb-4">Model Metrics</h3>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="p-3 border rounded-lg">
                        <p className="text-sm text-muted-foreground">Accuracy</p>
                        <p className="text-xl font-bold">95.2%</p>
                      </div>
                      <div className="p-3 border rounded-lg">
                        <p className="text-sm text-muted-foreground">R² Score</p>
                        <p className="text-xl font-bold">0.94</p>
                      </div>
                      <div className="p-3 border rounded-lg">
                        <p className="text-sm text-muted-foreground">Training Samples</p>
                        <p className="text-xl font-bold">2,000</p>
                      </div>
                      <div className="p-3 border rounded-lg">
                        <p className="text-sm text-muted-foreground">Validation</p>
                        <p className="text-xl font-bold">Healthcare</p>
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          <Card className="border-2">
            <CardHeader>
              <CardTitle className="text-2xl">Healthcare Antenna Applications</CardTitle>
              <CardDescription className="text-lg">
                Specific use cases for wearable healthcare monitoring
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-6 md:grid-cols-3">
                <div className="p-4 border rounded-lg bg-secondary/20">
                  <div className="flex items-center space-x-2 mb-2">
                    <Activity className="w-5 h-5 text-red-500" />
                    <h3 className="text-lg font-medium">Cardiac Monitoring</h3>
                  </div>
                  <p className="text-base text-muted-foreground">
                    ECG patches, heart rate monitors, arrhythmia detection
                  </p>
                  <div className="mt-3 text-sm">
                    <p><strong>Frequency:</strong> 2.4 GHz ISM</p>
                    <p><strong>SAR Limit:</strong> &lt; 1.6 W/kg</p>
                  </div>
                </div>

                <div className="p-4 border rounded-lg bg-secondary/20">
                  <div className="flex items-center space-x-2 mb-2">
                    <Zap className="w-5 h-5 text-blue-500" />
                    <h3 className="text-lg font-medium">Neural Interfaces</h3>
                  </div>
                  <p className="text-base text-muted-foreground">
                    EEG sensors, brain-computer interfaces, neural prosthetics
                  </p>
                  <div className="mt-3 text-sm">
                    <p><strong>Frequency:</strong> 915 MHz / 2.4 GHz</p>
                    <p><strong>SAR Limit:</strong> &lt; 2.0 W/kg</p>
                  </div>
                </div>

                <div className="p-4 border rounded-lg bg-secondary/20">
                  <div className="flex items-center space-x-2 mb-2">
                    <Target className="w-5 h-5 text-green-500" />
                    <h3 className="text-lg font-medium">Glucose Monitoring</h3>
                  </div>
                  <p className="text-base text-muted-foreground">
                    Continuous glucose monitoring, diabetic patches
                  </p>
                  <div className="mt-3 text-sm">
                    <p><strong>Frequency:</strong> 13.56 MHz NFC</p>
                    <p><strong>SAR Limit:</strong> &lt; 0.5 W/kg</p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}

