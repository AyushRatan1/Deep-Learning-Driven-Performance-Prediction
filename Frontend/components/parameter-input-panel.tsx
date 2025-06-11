"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import { SidebarGroup, SidebarGroupContent, SidebarGroupLabel } from "@/components/ui/sidebar"
import { useSAR, type FrequencyBand } from "@/components/sar-context"
import { sarAPI } from "@/lib/sar-api"
import { Shuffle, Play, Loader2, Settings2 } from "lucide-react"

const frequencyBands: FrequencyBand[] = [
  { id: "x-band", name: "X-band", range: "8-12 GHz", frequency: 10, color: "#3b82f6" },
  { id: "ku-band", name: "Ku-band", range: "12-18 GHz", frequency: 15, color: "#8b5cf6" },
  { id: "k-band", name: "K-band", range: "18-27 GHz", frequency: 22.5, color: "#06b6d4" },
  { id: "ka-band", name: "Ka-band", range: "27-40 GHz", frequency: 33.5, color: "#10b981" },
  { id: "v-band", name: "V-band", range: "40-75 GHz", frequency: 57.5, color: "#f59e0b" },
  { id: "w-band", name: "W-band", range: "75-110 GHz", frequency: 92.5, color: "#ef4444" },
  { id: "d-band", name: "D-band", range: "110-170 GHz", frequency: 140, color: "#ec4899" },
]

export function ParameterInputPanel() {
  const { state, dispatch } = useSAR()
  const [isGenerating, setIsGenerating] = useState(false)

  const handleBandChange = (bandId: string) => {
    const band = frequencyBands.find((b) => b.id === bandId)
    if (band) {
      dispatch({ type: "SET_BAND", payload: band })
    }
  }

  const handleParameterChange = (param: string, value: number[]) => {
    dispatch({ type: "SET_PARAMETERS", payload: { [param]: value[0] } })
  }

  const generateRandomParameters = () => {
    const randomParams = {
      substrateThickness: Math.random() * 3 + 0.5,
      permittivity: Math.random() * 6 + 2,
      patchWidth: Math.random() * 15 + 5,
      patchLength: Math.random() * 15 + 5,
      feedPosition: Math.random() * 0.4 + 0.1,
    }
    dispatch({ type: "SET_PARAMETERS", payload: randomParams })
  }

  const handlePredict = async () => {
    if (!state.selectedBand) return

    setIsGenerating(true)
    dispatch({ type: "SET_LOADING", payload: true })

    try {
      const prediction = await sarAPI.predict(state.selectedBand.id, state.parameters)
      dispatch({ type: "SET_PREDICTION", payload: prediction })
      dispatch({ type: "ADD_TO_HISTORY", payload: prediction })
    } catch (error) {
      console.error("Prediction failed:", error)
    } finally {
      setIsGenerating(false)
      dispatch({ type: "SET_LOADING", payload: false })
    }
  }

  return (
    <SidebarGroup className="px-0">
      <SidebarGroupLabel className="text-xs text-muted-foreground font-semibold flex items-center gap-2 px-2 mb-2">
        <Settings2 className="h-3 w-3" />
        Parameters
      </SidebarGroupLabel>
      <SidebarGroupContent>
        <Card className="border-border/50 bg-card/80 backdrop-blur-sm shadow-sm mx-2">
          <CardHeader className="pb-2 px-3 pt-3">
            <CardTitle className="text-xs flex items-center gap-2 font-semibold">
              <div className="h-1.5 w-1.5 rounded-full bg-gradient-to-r from-primary to-chart-2 animate-pulse"></div>
              Configuration
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 px-3 pb-3">
            {/* Frequency Band Selection */}
            <div className="space-y-1.5">
              <Label htmlFor="frequency-band" className="text-xs font-semibold">
                Frequency Band
              </Label>
              <Select onValueChange={handleBandChange} value={state.selectedBand?.id || ""}>
                <SelectTrigger className="bg-background/80 border-border/50 hover:border-primary/50 transition-colors h-8 text-xs">
                  <SelectValue placeholder="Select band" />
                </SelectTrigger>
                <SelectContent>
                  {frequencyBands.map((band) => (
                    <SelectItem key={band.id} value={band.id}>
                      <div className="flex items-center gap-2">
                        <div className="h-2 w-2 rounded-full shadow-sm" style={{ backgroundColor: band.color }} />
                        <span className="font-medium text-xs">{band.name}</span>
                        <span className="text-muted-foreground text-xs">({band.range})</span>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Input Mode Toggle */}
            <div className="flex items-center space-x-2 p-2 rounded-lg bg-gradient-to-r from-muted/50 to-muted/30 border border-border/30">
              <Switch
                id="input-mode"
                checked={state.inputMode === "custom"}
                onCheckedChange={(checked) =>
                  dispatch({ type: "SET_INPUT_MODE", payload: checked ? "custom" : "random" })
                }
                className="scale-75"
              />
              <Label htmlFor="input-mode" className="text-xs font-medium cursor-pointer">
                Custom Parameters
              </Label>
            </div>

            {/* Custom Parameters */}
            {state.inputMode === "custom" && (
              <div className="space-y-3 p-2 rounded-lg bg-gradient-to-br from-muted/20 to-muted/10 border border-border/20">
                <div className="space-y-2">
                  <Label className="text-xs font-semibold text-foreground">
                    Substrate: {state.parameters.substrateThickness.toFixed(1)}mm
                  </Label>
                  <Slider
                    value={[state.parameters.substrateThickness]}
                    onValueChange={(value) => handleParameterChange("substrateThickness", value)}
                    max={5}
                    min={0.1}
                    step={0.1}
                    className="w-full"
                  />
                </div>

                <div className="space-y-2">
                  <Label className="text-xs font-semibold text-foreground">
                    Permittivity: {state.parameters.permittivity.toFixed(1)}
                  </Label>
                  <Slider
                    value={[state.parameters.permittivity]}
                    onValueChange={(value) => handleParameterChange("permittivity", value)}
                    max={12}
                    min={1}
                    step={0.1}
                    className="w-full"
                  />
                </div>

                <div className="space-y-2">
                  <Label className="text-xs font-semibold text-foreground">
                    Width: {state.parameters.patchWidth.toFixed(1)}mm
                  </Label>
                  <Slider
                    value={[state.parameters.patchWidth]}
                    onValueChange={(value) => handleParameterChange("patchWidth", value)}
                    max={25}
                    min={2}
                    step={0.1}
                    className="w-full"
                  />
                </div>

                <div className="space-y-2">
                  <Label className="text-xs font-semibold text-foreground">
                    Length: {state.parameters.patchLength.toFixed(1)}mm
                  </Label>
                  <Slider
                    value={[state.parameters.patchLength]}
                    onValueChange={(value) => handleParameterChange("patchLength", value)}
                    max={25}
                    min={2}
                    step={0.1}
                    className="w-full"
                  />
                </div>

                <div className="space-y-2">
                  <Label className="text-xs font-semibold text-foreground">
                    Feed: {state.parameters.feedPosition.toFixed(2)}
                  </Label>
                  <Slider
                    value={[state.parameters.feedPosition]}
                    onValueChange={(value) => handleParameterChange("feedPosition", value)}
                    max={0.5}
                    min={0.05}
                    step={0.01}
                    className="w-full"
                  />
                </div>
              </div>
            )}

            {/* Random Parameters Button */}
            {state.inputMode === "random" && (
              <Button
                onClick={generateRandomParameters}
                variant="outline"
                size="sm"
                className="w-full bg-gradient-to-r from-muted/50 to-muted/30 hover:from-muted/70 hover:to-muted/50 border-border/50 transition-all duration-200 h-8 text-xs"
              >
                <Shuffle className="h-3 w-3 mr-1.5" />
                Random Parameters
              </Button>
            )}

            {/* Predict Button */}
            <Button
              onClick={handlePredict}
              className="w-full bg-gradient-to-r from-primary to-chart-2 hover:from-primary/90 hover:to-chart-2/90 text-primary-foreground font-semibold shadow-lg hover:shadow-xl transition-all duration-200 h-8 text-xs"
              disabled={!state.selectedBand || isGenerating}
              size="sm"
            >
              {isGenerating ? (
                <>
                  <Loader2 className="h-3 w-3 mr-1.5 animate-spin" />
                  Predicting...
                </>
              ) : (
                <>
                  <Play className="h-3 w-3 mr-1.5" />
                  Run Prediction
                </>
              )}
            </Button>
          </CardContent>
        </Card>
      </SidebarGroupContent>
    </SidebarGroup>
  )
}
