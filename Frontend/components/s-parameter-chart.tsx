"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from "recharts"
import { Download, Maximize2 } from "lucide-react"
import { useSAR } from "@/components/sar-context"

export function SParameterChart() {
  const { state } = useSAR()
  const { currentPrediction } = state

  const handleDownload = () => {
    if (!currentPrediction) return

    const csvContent =
      "data:text/csv;charset=utf-8," +
      "Frequency (GHz),S11 (dB)\n" +
      currentPrediction.sParameters
        .map((point) => `${(point.frequency / 1000000000).toFixed(3)},${point.s11.toFixed(3)}`)
        .join("\n")

    const encodedUri = encodeURI(csvContent)
    const link = document.createElement("a")
    link.setAttribute("href", encodedUri)
    link.setAttribute("download", `s_parameters_${currentPrediction.id}.csv`)
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  const handleMaximize = () => {
    // This would open a full-screen chart view
    console.log("Maximize chart view")
  }

  if (!currentPrediction || !currentPrediction.sParameters) {
    return (
      <Card className="border-border/50 bg-gradient-to-br from-card to-muted/20">
        <CardHeader>
          <CardTitle className="text-sm font-medium">S-Parameters (Return Loss)</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-64 flex items-center justify-center text-muted-foreground">
            <div className="text-center">
              <div className="h-16 w-16 mx-auto mb-3 rounded-full bg-muted/50 flex items-center justify-center">📊</div>
              <p>No S-parameter data available</p>
            </div>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className="border-border/50 bg-gradient-to-br from-card to-muted/20">
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="text-sm font-medium">S-Parameters (Return Loss)</CardTitle>
        <div className="flex gap-2">
          <Button size="sm" variant="ghost" onClick={handleDownload}>
            <Download className="h-4 w-4" />
          </Button>
          <Button size="sm" variant="ghost" onClick={handleMaximize}>
            <Maximize2 className="h-4 w-4" />
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={currentPrediction.sParameters}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
              <XAxis
                dataKey="frequency"
                tickFormatter={(value) => `${(value / 1000000000).toFixed(1)}G`}
                stroke="hsl(var(--muted-foreground))"
              />
              <YAxis domain={[-40, 0]} tickFormatter={(value) => `${value}dB`} stroke="hsl(var(--muted-foreground))" />
              <Tooltip
                formatter={(value: number) => [`${value.toFixed(2)} dB`, "S11"]}
                labelFormatter={(value) => `Frequency: ${(value / 1000000000).toFixed(2)} GHz`}
                contentStyle={{
                  backgroundColor: "hsl(var(--card))",
                  border: "1px solid hsl(var(--border))",
                  borderRadius: "8px",
                }}
              />
              <ReferenceLine y={-10} stroke="#ef4444" strokeDasharray="5 5" opacity={0.7} />
              <Line
                type="monotone"
                dataKey="s11"
                stroke="url(#colorGradient)"
                strokeWidth={3}
                dot={false}
                activeDot={{ r: 6, fill: "hsl(var(--primary))" }}
              />
              <defs>
                <linearGradient id="colorGradient" x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0%" stopColor="hsl(var(--primary))" />
                  <stop offset="100%" stopColor="hsl(var(--chart-2))" />
                </linearGradient>
              </defs>
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div className="text-xs text-muted-foreground mt-2 flex items-center justify-between">
          <span>Red line indicates -10dB threshold</span>
          <span>{currentPrediction.sParameters.length} data points</span>
        </div>
      </CardContent>
    </Card>
  )
}
