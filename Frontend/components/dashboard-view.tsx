"use client"

import { SARDisplay } from "@/components/sar-display"
import { SParameterChart } from "@/components/s-parameter-chart"
import { RadiationPattern3D } from "@/components/radiation-pattern-3d"
import { PerformanceMetrics } from "@/components/performance-metrics"
import { useSAR } from "@/components/sar-context"
import { useSARApi } from "@/hooks/use-sar-api"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Zap, Activity } from "lucide-react"

export function DashboardView() {
  const { state } = useSAR()
  const { predictSAR, generateSampleParameters, isLoading } = useSARApi()

  const handlePredictSAR = async () => {
    if (!state.selectedBand) return
    try {
      await predictSAR()
    } catch (error) {
      console.error("Prediction failed:", error)
    }
  }

  const handleGenerateSample = async () => {
    if (!state.selectedBand) return
    try {
      await generateSampleParameters()
    } catch (error) {
      console.error("Sample generation failed:", error)
    }
  }

  return (
    <div className="space-y-6 p-6">
      {!state.currentPrediction && (
        <div className="text-center py-12">
          <div className="max-w-md mx-auto">
            <div className="h-24 w-24 mx-auto mb-4 rounded-full bg-gradient-to-br from-primary/20 to-chart-2/20 flex items-center justify-center">
              <div className="h-12 w-12 rounded-full bg-gradient-to-br from-primary to-chart-2 flex items-center justify-center">
                <span className="text-2xl">📡</span>
              </div>
            </div>
            <h3 className="text-lg font-semibold mb-2">Welcome to SAR Predictor Pro</h3>
            <p className="text-muted-foreground mb-4">
              Select a frequency band and configure parameters to start predicting SAR values for your textile antenna
              designs.
            </p>
            
            {state.selectedBand && (
              <div className="space-y-4">
                <Badge variant="secondary" className="mb-2">
                  {state.selectedBand.name} Selected
                </Badge>
                <div className="flex gap-2 justify-center">
                  <Button onClick={handleGenerateSample} variant="outline" disabled={isLoading}>
                    Generate Sample
                  </Button>
                  <Button onClick={handlePredictSAR} disabled={isLoading}>
                    <Zap className="w-4 h-4 mr-2" />
                    Predict SAR
                  </Button>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {state.currentPrediction && (
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          <div className="space-y-6">
            <SARDisplay prediction={state.currentPrediction} />
            <PerformanceMetrics prediction={state.currentPrediction} />
          </div>
          
          <div className="space-y-6">
            <SParameterChart prediction={state.currentPrediction} />
            <RadiationPattern3D prediction={state.currentPrediction} />
          </div>
        </div>
      )}

      {/* Quick Actions */}
      {state.currentPrediction && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="w-5 h-5" />
              Quick Actions
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex gap-2">
              <Button onClick={handleGenerateSample} variant="outline" disabled={isLoading}>
                New Sample
              </Button>
              <Button onClick={handlePredictSAR} disabled={isLoading}>
                <Zap className="w-4 h-4 mr-2" />
                Re-predict
              </Button>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
