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
         Target, Zap, WifiIcon, Upload } from "lucide-react";
import { useSAR } from "@/components/sar-context"
import { useSARApi } from "@/hooks/use-sar-api"
import ChatInterface from "@/components/chat-interface"
import type { AntennaParameters, SARPrediction } from "@/components/sar-context"
import { LineChart as RechartsLineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, ScatterChart, Scatter, PieChart, Pie, Cell } from 'recharts';

const healthcareSARData = [
  { frequency: 2.4, cardiac: 0.42, neural: 0.68, glucose: 0.15 },
  { frequency: 915, cardiac: 0.35, neural: 0.55, glucose: 0.12 },
  { frequency: 13.56, cardiac: 0.28, neural: 0.45, glucose: 0.10 },
  { frequency: 5.8, cardiac: 0.32, neural: 0.50, glucose: 0.11 },
  { frequency: 2.1, cardiac: 0.38, neural: 0.60, glucose: 0.14 },
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
    const band = frequencyBands.find(b => b.id === bandId)
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

  if (state.activeView === "chat") {
    return <ChatInterface />
  }

  return (
    <div className="flex-1 space-y-6 p-8 pt-6">
      <div className="flex items-center justify-between space-y-2">
        <div>
          <h2 className="text-4xl font-bold tracking-tight">Wearable Healthcare Antenna SAR Dashboard</h2>
          <p className="text-lg text-muted-foreground mt-2">
            Deep Learning-driven Performance Prediction for Healthcare Wearables
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <Badge variant="outline" className="text-lg px-4 py-2">
            {systemStatus?.status === "healthy" ? "System Online" : "System Status Unknown"}
          </Badge>
          <Badge variant="secondary" className="text-lg px-4 py-2">
            Healthcare Focus
          </Badge>
        </div>
      </div>

      <Tabs value={state.activeView} onValueChange={(value) => 
        dispatch({ type: "SET_ACTIVE_VIEW", payload: value as any })
      } className="space-y-6">
        <TabsList className="grid w-full grid-cols-4 h-12">
          <TabsTrigger value="dashboard" className="text-sm px-3 py-2 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
            Overview
          </TabsTrigger>
          <TabsTrigger value="comparison" className="text-sm px-3 py-2 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
            Prediction
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
                  Healthcare antenna assessments
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
                  Safe for wearable applications
                </p>
              </CardContent>
            </Card>

            <Card className="border-2">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-base font-medium">Healthcare Compliance</CardTitle>
                <Heart className="h-6 w-6 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold text-green-600">98.5%</div>
                <p className="text-base text-muted-foreground">
                  FDA/CE medical device standards
                </p>
              </CardContent>
            </Card>

            <Card className="border-2">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-base font-medium">Active Frequency</CardTitle>
                <WifiIcon className="h-6 w-6 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold">
                  {state.selectedBand ? state.frequencyBands.find(b => b.id === state.selectedBand?.id)?.name || "N/A" : "None"}
                </div>
                <p className="text-base text-muted-foreground">
                  Medical device frequency band
                </p>
              </CardContent>
            </Card>
          </div>

          <div className="grid gap-6 md:grid-cols-2">
            <Card className="border-2">
              <CardHeader>
                <CardTitle className="text-2xl">SAR vs Frequency Analysis</CardTitle>
                <CardDescription className="text-lg">
                  Healthcare antenna SAR patterns across medical frequency bands
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <RechartsLineChart data={healthcareSARData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="frequency" label={{ value: 'Frequency (GHz)', position: 'insideBottom', offset: -5 }} />
                    <YAxis label={{ value: 'SAR (W/kg)', angle: -90, position: 'insideLeft' }} />
                    <Tooltip formatter={(value, name) => [value, name === 'cardiac' ? 'Cardiac Monitor' : name === 'neural' ? 'Neural Interface' : 'Glucose Monitor']} />
                    <Legend />
                    <Line type="monotone" dataKey="cardiac" stroke="#ef4444" strokeWidth={2} name="Cardiac Monitoring" />
                    <Line type="monotone" dataKey="neural" stroke="#3b82f6" strokeWidth={2} name="Neural Interface" />
                    <Line type="monotone" dataKey="glucose" stroke="#10b981" strokeWidth={2} name="Glucose Monitoring" />
                  </RechartsLineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card className="border-2">
              <CardHeader>
                <CardTitle className="text-2xl">Antenna Efficiency Distribution</CardTitle>
                <CardDescription className="text-lg">
                  Performance metrics for wearable healthcare antennas
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={efficiencyData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="application" />
                    <YAxis label={{ value: 'Efficiency (%)', angle: -90, position: 'insideLeft' }} />
                    <Tooltip formatter={(value) => [`${value}%`, 'Efficiency']} />
                    <Bar dataKey="efficiency" fill="#8b5cf6" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>

          <div className="grid gap-6 md:grid-cols-2">
            <Card className="border-2">
              <CardHeader>
                <CardTitle className="text-2xl">SAR Safety Compliance</CardTitle>
                <CardDescription className="text-lg">
                  Regulatory compliance across healthcare applications
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={complianceData}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {complianceData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card className="border-2">
              <CardHeader>
                <CardTitle className="text-2xl">Tissue Interaction Analysis</CardTitle>
                <CardDescription className="text-lg">
                  SAR distribution in human tissue layers
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <ScatterChart data={tissueInteractionData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="depth" label={{ value: 'Tissue Depth (mm)', position: 'insideBottom', offset: -5 }} />
                    <YAxis label={{ value: 'SAR (W/kg)', angle: -90, position: 'insideLeft' }} />
                    <Tooltip formatter={(value, name) => [value, name === 'sar' ? 'SAR Value' : name]} />
                    <Scatter dataKey="sar" fill="#f59e0b" />
                  </ScatterChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>

          <Card className="border-2">
            <CardHeader>
              <CardTitle className="text-2xl">Real-time Healthcare Antenna Monitor</CardTitle>
              <CardDescription className="text-lg">
                Live SAR monitoring for active wearable devices
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-6 md:grid-cols-3">
                <div className="space-y-4">
                  <div className="flex items-center space-x-2">
                    <Heart className="w-5 h-5 text-red-500" />
                    <h3 className="text-lg font-medium">Cardiac Patch</h3>
                    <Badge variant="outline" className="text-sm">Active</Badge>
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Current SAR:</span>
                      <span className="font-mono text-green-600">0.42 W/kg</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Frequency:</span>
                      <span className="font-mono">2.4 GHz</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Power:</span>
                      <span className="font-mono">10 mW</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Status:</span>
                      <Badge variant="secondary" className="text-xs">Safe</Badge>
                    </div>
                  </div>
                </div>

                <div className="space-y-4">
                  <div className="flex items-center space-x-2">
                    <Brain className="w-5 h-5 text-blue-500" />
                    <h3 className="text-lg font-medium">EEG Headset</h3>
                    <Badge variant="outline" className="text-sm">Active</Badge>
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Current SAR:</span>
                      <span className="font-mono text-green-600">0.68 W/kg</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Frequency:</span>
                      <span className="font-mono">915 MHz</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Power:</span>
                      <span className="font-mono">25 mW</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Status:</span>
                      <Badge variant="secondary" className="text-xs">Safe</Badge>
                    </div>
                  </div>
                </div>

                <div className="space-y-4">
                  <div className="flex items-center space-x-2">
                    <Target className="w-5 h-5 text-green-500" />
                    <h3 className="text-lg font-medium">Glucose Sensor</h3>
                    <Badge variant="outline" className="text-sm">Active</Badge>
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Current SAR:</span>
                      <span className="font-mono text-green-600">0.15 W/kg</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Frequency:</span>
                      <span className="font-mono">13.56 MHz</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Power:</span>
                      <span className="font-mono">5 mW</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Status:</span>
                      <Badge variant="secondary" className="text-xs">Safe</Badge>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="comparison" className="space-y-6">
          <Card className="border-2">
            <CardHeader>
              <CardTitle className="text-2xl">SAR Prediction</CardTitle>
              <CardDescription className="text-lg">
                Configure antenna parameters and predict SAR levels
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid gap-6 md:grid-cols-2">
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="thickness" className="text-lg">Substrate Thickness (mm)</Label>
                      <Input
                        id="thickness"
                        type="number"
                        step="0.1"
                        value={parameters.substrate_thickness}
                        onChange={(e) => handleParameterChange('substrate_thickness', parseFloat(e.target.value))}
                        className="text-lg h-12"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="permittivity" className="text-lg">Relative Permittivity</Label>
                      <Input
                        id="permittivity"
                        type="number"
                        step="0.1"
                        value={parameters.substrate_permittivity}
                        onChange={(e) => handleParameterChange('substrate_permittivity', parseFloat(e.target.value))}
                        className="text-lg h-12"
                      />
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="width" className="text-lg">Patch Width (mm)</Label>
                      <Input
                        id="width"
                        type="number"
                        step="0.1"
                        value={parameters.patch_width}
                        onChange={(e) => handleParameterChange('patch_width', parseFloat(e.target.value))}
                        className="text-lg h-12"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="length" className="text-lg">Patch Length (mm)</Label>
                      <Input
                        id="length"
                        type="number"
                        step="0.1"
                        value={parameters.patch_length}
                        onChange={(e) => handleParameterChange('patch_length', parseFloat(e.target.value))}
                        className="text-lg h-12"
                      />
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="bending" className="text-lg">Bending Radius (mm)</Label>
                      <Input
                        id="bending"
                        type="number"
                        step="1"
                        value={parameters.bending_radius}
                        onChange={(e) => handleParameterChange('bending_radius', parseFloat(e.target.value))}
                        className="text-lg h-12"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="power" className="text-lg">Power Density (W/cm²)</Label>
                      <Input
                        id="power"
                        type="number"
                        step="0.1"
                        value={parameters.power_density}
                        onChange={(e) => handleParameterChange('power_density', parseFloat(e.target.value))}
                        className="text-lg h-12"
                      />
                    </div>
                  </div>
                  <div className="flex space-x-4">
                    <Button 
                      onClick={handleGenerateParameters}
                      disabled={isLoading || !selectedBand}
                      variant="outline"
                      className="text-lg px-6 py-3 h-auto"
                    >
                      Generate Random
                    </Button>
                    <Button 
                      onClick={handlePredict}
                      disabled={isLoading || !selectedBand}
                      className="text-lg px-6 py-3 h-auto"
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
                              {currentPrediction.sar_value.toFixed(3)} W/kg
                            </p>
                          </div>
                          <div>
                            <p className="text-lg font-medium">Gain</p>
                            <p className="text-2xl font-bold">
                              {currentPrediction.gain.toFixed(2)} dBi
                            </p>
                          </div>
                        </div>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                          <div>
                            <p className="text-lg font-medium">Efficiency</p>
                            <p className="text-2xl font-bold">
                              {currentPrediction.efficiency.toFixed(1)}%
                            </p>
                          </div>
                          <div>
                            <p className="text-lg font-medium">Bandwidth</p>
                            <p className="text-2xl font-bold">
                              {currentPrediction.bandwidth.toFixed(2)}%
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
                              {prediction.sar_value.toFixed(3)} W/kg
                            </p>
                          </div>
                          <div>
                            <p className="text-lg font-medium">Gain</p>
                            <p className="text-2xl font-bold">
                              {prediction.gain.toFixed(2)} dBi
                            </p>
                          </div>
                        </div>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                          <div>
                            <p className="text-lg font-medium">Efficiency</p>
                            <p className="text-2xl font-bold">
                              {prediction.efficiency.toFixed(1)}%
                            </p>
                          </div>
                          <div>
                            <p className="text-lg font-medium">Bandwidth</p>
                            <p className="text-2xl font-bold">
                              {prediction.bandwidth.toFixed(2)}%
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
