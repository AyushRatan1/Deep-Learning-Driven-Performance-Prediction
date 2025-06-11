"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { TrendingUp, Zap, Radio, Target } from "lucide-react"
import { useSAR } from "@/components/sar-context"

export function PerformanceMetrics() {
  const { state } = useSAR()
  const { currentPrediction } = state

  if (!currentPrediction) {
    return (
      <Card className="border-border/50 bg-gradient-to-br from-card to-muted/20">
        <CardHeader>
          <CardTitle className="text-sm font-semibold flex items-center gap-2 text-bright">
            <Target className="h-4 w-4 text-primary" />
            Performance Metrics
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center text-bright-muted font-medium py-8">
            <p>No prediction data available</p>
          </div>
        </CardContent>
      </Card>
    )
  }

  const metrics = [
    {
      title: "Gain",
      value: `${currentPrediction.gain.toFixed(1)} dBi`,
      icon: TrendingUp,
      description: "Peak directional gain",
      color: "from-blue-500 to-cyan-500",
      bgColor: "from-blue-500/10 to-cyan-500/10",
    },
    {
      title: "Efficiency",
      value: `${(currentPrediction.efficiency * 100).toFixed(1)}%`,
      icon: Zap,
      description: "Radiation efficiency",
      color: "from-emerald-500 to-green-500",
      bgColor: "from-emerald-500/10 to-green-500/10",
    },
    {
      title: "Bandwidth",
      value: `${currentPrediction.bandwidth.toFixed(1)} MHz`,
      icon: Radio,
      description: "-10dB return loss BW",
      color: "from-purple-500 to-pink-500",
      bgColor: "from-purple-500/10 to-pink-500/10",
    },
  ]

  return (
    <Card className="border-border/50 bg-gradient-to-br from-card to-muted/20">
      <CardHeader>
        <CardTitle className="text-sm font-semibold flex items-center gap-2 text-bright">
          <Target className="h-4 w-4 text-primary" />
          Performance Metrics
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-3 gap-4">
          {metrics.map((metric, index) => {
            const Icon = metric.icon
            return (
              <div
                key={index}
                className={`text-center space-y-3 p-4 rounded-lg bg-gradient-to-br ${metric.bgColor} border border-border/30`}
              >
                <div
                  className={`h-10 w-10 mx-auto rounded-full bg-gradient-to-br ${metric.color} flex items-center justify-center`}
                >
                  <Icon className="h-5 w-5 text-white" />
                </div>
                <div className="space-y-1">
                  <div className="text-2xl font-bold">{metric.value}</div>
                  <div className="text-xs text-bright-muted font-medium">{metric.description}</div>
                </div>
              </div>
            )
          })}
        </div>
      </CardContent>
    </Card>
  )
}
