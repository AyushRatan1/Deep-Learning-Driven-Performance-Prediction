"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { AlertTriangle, CheckCircle, XCircle, Shield, Map } from "lucide-react"
import { useSAR } from "@/components/sar-context"
import { CircularSARMap } from "@/components/circular-sar-map"

export function SARDisplay() {
  const { state } = useSAR()
  const { currentPrediction } = state

  if (!currentPrediction) {
    return (
      <Card className="border-border/50 bg-gradient-to-br from-card to-muted/20">
        <CardHeader>
          <CardTitle className="text-sm font-semibold flex items-center gap-2 text-bright">
            <Shield className="h-4 w-4 text-primary" />
            SAR Value
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center text-muted-foreground">
            <div className="h-16 w-16 mx-auto mb-3 rounded-full bg-muted/50 flex items-center justify-center">
              <Shield className="h-8 w-8 text-muted-foreground/50" />
            </div>
            <p className="text-sm text-bright-muted font-medium">No prediction available</p>
            <p className="text-xs mt-1 text-bright-muted">Run a prediction to see SAR values</p>
          </div>
        </CardContent>
      </Card>
    )
  }

  const sarValue = isNaN(currentPrediction.sarValue) ? 0 : currentPrediction.sarValue
  const getSafetyStatus = (sar: number) => {
    if (sar <= 1.6)
      return {
        status: "safe",
        color: "emerald",
        icon: CheckCircle,
        bgColor: "from-emerald-500/10 to-green-500/10",
        textColor: "text-emerald-600 dark:text-emerald-400",
      }
    if (sar <= 2.0)
      return {
        status: "warning",
        color: "amber",
        icon: AlertTriangle,
        bgColor: "from-amber-500/10 to-yellow-500/10",
        textColor: "text-amber-600 dark:text-amber-400",
      }
    return {
      status: "unsafe",
      color: "red",
      icon: XCircle,
      bgColor: "from-red-500/10 to-rose-500/10",
      textColor: "text-red-600 dark:text-red-400",
    }
  }

  const safety = getSafetyStatus(sarValue)
  const SafetyIcon = safety.icon
  const progressValue = Math.min((sarValue / 2.0) * 100, 100)

  return (
    <Card className={`border-border/50 bg-gradient-to-br ${safety.bgColor} backdrop-blur`}>
      <CardHeader>
        <CardTitle className="text-sm font-semibold flex items-center gap-2 text-bright">
          <Shield className="h-4 w-4 text-primary" />
          SAR Analysis
        </CardTitle>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="value" className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="value" className="text-xs">
              <Shield className="h-3 w-3 mr-1" />
              SAR Value
            </TabsTrigger>
            <TabsTrigger value="map" className="text-xs">
              <Map className="h-3 w-3 mr-1" />
              Spatial Map
            </TabsTrigger>
          </TabsList>

          <TabsContent value="value" className="mt-4">
        <div className="text-center space-y-4">
          <div className="relative">
            <div className={`text-4xl font-bold ${safety.textColor}`}>{sarValue.toFixed(3)}</div>
            <div className="text-sm text-bright-muted font-medium">W/kg</div>
          </div>

          <div className="space-y-2">
            <Progress
              value={progressValue}
              className="h-2"
              style={{
                background: `linear-gradient(to right, 
                  ${progressValue <= 80 ? "#10b981" : progressValue <= 100 ? "#f59e0b" : "#ef4444"} 0%, 
                  ${progressValue <= 80 ? "#10b981" : progressValue <= 100 ? "#f59e0b" : "#ef4444"} ${progressValue}%, 
                  hsl(var(--muted)) ${progressValue}%)`,
              }}
            />
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>0</span>
              <span>1.6</span>
              <span>2.0</span>
            </div>
          </div>

          <div className="flex items-center justify-center gap-2">
            <SafetyIcon className={`h-5 w-5 ${safety.textColor}`} />
            <Badge
              variant={safety.status === "safe" ? "default" : safety.status === "warning" ? "secondary" : "destructive"}
              className="font-medium"
            >
              {safety.status.toUpperCase()}
            </Badge>
          </div>

          <div className="text-xs text-bright-muted p-2 rounded bg-muted/30 font-medium">
            FCC/ICNIRP Limit: 1.6 W/kg
          </div>
        </div>
          </TabsContent>

          <TabsContent value="map" className="mt-4">
            <CircularSARMap 
              sarData={currentPrediction}
              showSafetyZones={true}
              animationSpeed={2}
            />
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}
