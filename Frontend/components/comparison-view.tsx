"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line } from "recharts"
import { useSAR } from "@/components/sar-context"
import { sarAPI } from "@/lib/sar-api"
import { useState } from "react"
import { RefreshCw, Download, TrendingUp } from "lucide-react"

export function ComparisonView() {
  const { state, dispatch } = useSAR()
  const [isLoading, setIsLoading] = useState(false)

  const handleGenerateComparison = async () => {
    setIsLoading(true)
    try {
      const comparisonData = await sarAPI.getComparisonData()
      dispatch({ type: "SET_COMPARISON_DATA", payload: comparisonData })
    } catch (error) {
      console.error("Failed to generate comparison:", error)
    } finally {
      setIsLoading(false)
    }
  }

  const handleDownloadComparison = () => {
    if (state.comparisonData.length === 0) return

    const csvContent =
      "data:text/csv;charset=utf-8," +
      "Band,SAR (W/kg),Gain (dBi),Efficiency (%),Bandwidth (MHz)\n" +
      state.comparisonData
        .map(
          (prediction) =>
            `${prediction.bandId.toUpperCase()},${prediction.sarValue.toFixed(3)},${prediction.gain.toFixed(1)},${(prediction.efficiency * 100).toFixed(1)},${prediction.bandwidth.toFixed(1)}`,
        )
        .join("\n")

    const encodedUri = encodeURI(csvContent)
    const link = document.createElement("a")
    link.setAttribute("href", encodedUri)
    link.setAttribute("download", "frequency_band_comparison.csv")
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  const chartData = state.comparisonData.map((prediction) => ({
    band: prediction.bandId.toUpperCase().replace("-BAND", ""),
    sar: prediction.sarValue,
    gain: prediction.gain,
    efficiency: prediction.efficiency * 100,
    bandwidth: prediction.bandwidth,
  }))

  return (
    <div className="space-y-6 p-6">
      <Card className="border-border/50 bg-gradient-to-br from-card to-muted/20">
        <CardHeader className="flex flex-row items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5 text-primary" />
              Frequency Band Comparison
            </CardTitle>
            <p className="text-sm text-muted-foreground mt-1">
              Compare SAR and performance metrics across different frequency bands
            </p>
          </div>
          <div className="flex gap-2">
            {state.comparisonData.length > 0 && (
              <Button onClick={handleDownloadComparison} size="sm" variant="outline">
                <Download className="h-4 w-4 mr-2" />
                Export
              </Button>
            )}
            <Button
              onClick={handleGenerateComparison}
              disabled={isLoading}
              className="bg-gradient-to-r from-primary to-chart-2"
            >
              <RefreshCw className={`h-4 w-4 mr-2 ${isLoading ? "animate-spin" : ""}`} />
              {isLoading ? "Generating..." : "Generate Comparison"}
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {state.comparisonData.length === 0 ? (
            <div className="h-64 flex items-center justify-center text-muted-foreground">
              <div className="text-center">
                <div className="h-20 w-20 mx-auto mb-4 rounded-full bg-gradient-to-br from-primary/20 to-chart-2/20 flex items-center justify-center">
                  <TrendingUp className="h-10 w-10 text-primary" />
                </div>
                <p className="text-lg font-medium mb-2">No Comparison Data</p>
                <p className="text-sm">Click "Generate Comparison" to analyze SAR across frequency bands</p>
              </div>
            </div>
          ) : (
            <div className="space-y-6">
              {/* SAR Comparison Chart */}
              <div className="space-y-4">
                <h3 className="text-lg font-semibold">SAR Values by Frequency Band</h3>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
                      <XAxis dataKey="band" stroke="hsl(var(--muted-foreground))" />
                      <YAxis stroke="hsl(var(--muted-foreground))" />
                      <Tooltip
                        formatter={(value: number, name: string) => {
                          if (name === "sar") return [`${value.toFixed(3)} W/kg`, "SAR"]
                          return [value, name]
                        }}
                        contentStyle={{
                          backgroundColor: "hsl(var(--card))",
                          border: "1px solid hsl(var(--border))",
                          borderRadius: "8px",
                        }}
                      />
                      <Bar dataKey="sar" fill="url(#sarGradient)" radius={[4, 4, 0, 0]} />
                      <defs>
                        <linearGradient id="sarGradient" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="hsl(var(--primary))" />
                          <stop offset="100%" stopColor="hsl(var(--chart-2))" />
                        </linearGradient>
                      </defs>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Performance Metrics Chart */}
              <div className="space-y-4">
                <h3 className="text-lg font-semibold">Performance Metrics Comparison</h3>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
                      <XAxis dataKey="band" stroke="hsl(var(--muted-foreground))" />
                      <YAxis stroke="hsl(var(--muted-foreground))" />
                      <Tooltip
                        formatter={(value: number, name: string) => {
                          if (name === "gain") return [`${value.toFixed(1)} dBi`, "Gain"]
                          if (name === "efficiency") return [`${value.toFixed(1)}%`, "Efficiency"]
                          return [value, name]
                        }}
                        contentStyle={{
                          backgroundColor: "hsl(var(--card))",
                          border: "1px solid hsl(var(--border))",
                          borderRadius: "8px",
                        }}
                      />
                      <Line type="monotone" dataKey="gain" stroke="hsl(var(--chart-1))" strokeWidth={3} />
                      <Line type="monotone" dataKey="efficiency" stroke="hsl(var(--chart-3))" strokeWidth={3} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Summary Cards */}
              <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
                {state.comparisonData.map((prediction, index) => {
                  const getSafetyColor = (sar: number) => {
                    if (sar <= 1.6) return "from-emerald-500/10 to-green-500/10 border-emerald-500/20"
                    if (sar <= 2.0) return "from-amber-500/10 to-yellow-500/10 border-amber-500/20"
                    return "from-red-500/10 to-rose-500/10 border-red-500/20"
                  }

                  return (
                    <Card key={index} className={`bg-gradient-to-br ${getSafetyColor(prediction.sarValue)} border`}>
                      <CardContent className="p-4 text-center space-y-2">
                        <Badge variant="outline" className="mb-2">
                          {prediction.bandId.toUpperCase()}
                        </Badge>
                        <div className="space-y-1">
                          <div className="text-lg font-bold">{prediction.sarValue.toFixed(3)}</div>
                          <div className="text-xs text-muted-foreground">W/kg</div>
                        </div>
                        <div className="space-y-1 text-xs">
                          <div>Gain: {prediction.gain.toFixed(1)} dBi</div>
                          <div>Eff: {(prediction.efficiency * 100).toFixed(1)}%</div>
                          <div>BW: {prediction.bandwidth.toFixed(0)} MHz</div>
                        </div>
                        <Progress value={Math.min((prediction.sarValue / 2.0) * 100, 100)} className="h-1" />
                      </CardContent>
                    </Card>
                  )
                })}
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
