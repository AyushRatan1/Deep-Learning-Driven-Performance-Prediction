"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Eye, Download, Trash2, Search, Filter, History } from "lucide-react"
import { useSAR } from "@/components/sar-context"
import { useState } from "react"

export function PredictionHistory() {
  const { state, dispatch } = useSAR()
  const [searchTerm, setSearchTerm] = useState("")
  const [filterBand, setFilterBand] = useState("all")
  const [sortBy, setSortBy] = useState("timestamp")

  const getSafetyBadge = (sarValue: number) => {
    if (sarValue <= 1.6)
      return <Badge className="bg-emerald-100 text-emerald-800 dark:bg-emerald-900 dark:text-emerald-200">Safe</Badge>
    if (sarValue <= 2.0)
      return <Badge className="bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-200">Warning</Badge>
    return <Badge className="bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200">Unsafe</Badge>
  }

  const handleViewPrediction = (prediction: any) => {
    dispatch({ type: "SET_SELECTED_PREDICTION", payload: prediction })
    dispatch({ type: "SET_PREDICTION", payload: prediction })
    dispatch({ type: "SET_ACTIVE_VIEW", payload: "dashboard" })
  }

  const handleDownloadPrediction = (prediction: any) => {
    const data = {
      id: prediction.id,
      timestamp: prediction.timestamp,
      band: prediction.bandId,
      parameters: prediction.parameters,
      results: {
        sar: prediction.sarValue,
        gain: prediction.gain,
        efficiency: prediction.efficiency,
        bandwidth: prediction.bandwidth,
      },
    }

    const jsonContent = "data:text/json;charset=utf-8," + JSON.stringify(data, null, 2)
    const encodedUri = encodeURI(jsonContent)
    const link = document.createElement("a")
    link.setAttribute("href", encodedUri)
    link.setAttribute("download", `prediction_${prediction.id}.json`)
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  const handleDeletePrediction = (predictionId: string) => {
    // In a real app, this would call an API to delete the prediction
    const updatedHistory = state.predictionHistory.filter((p) => p.id !== predictionId)
    // We'd need to add a DELETE_PREDICTION action to the reducer
    console.log("Delete prediction:", predictionId)
  }

  const handleExportAll = () => {
    if (state.predictionHistory.length === 0) return

    const csvContent =
      "data:text/csv;charset=utf-8," +
      "ID,Timestamp,Band,SAR (W/kg),Gain (dBi),Efficiency (%),Bandwidth (MHz),Safety Status\n" +
      state.predictionHistory
        .map((prediction) => {
          const safetyStatus = prediction.sarValue <= 1.6 ? "Safe" : prediction.sarValue <= 2.0 ? "Warning" : "Unsafe"
          return `${prediction.id},${prediction.timestamp},${prediction.bandId.toUpperCase()},${prediction.sarValue.toFixed(3)},${prediction.gain.toFixed(1)},${(prediction.efficiency * 100).toFixed(1)},${prediction.bandwidth.toFixed(1)},${safetyStatus}`
        })
        .join("\n")

    const encodedUri = encodeURI(csvContent)
    const link = document.createElement("a")
    link.setAttribute("href", encodedUri)
    link.setAttribute("download", "prediction_history.csv")
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  // Filter and sort predictions
  const filteredPredictions = state.predictionHistory
    .filter((prediction) => {
      const matchesSearch =
        prediction.bandId.toLowerCase().includes(searchTerm.toLowerCase()) ||
        prediction.id.toLowerCase().includes(searchTerm.toLowerCase())
      const matchesBand = filterBand === "all" || prediction.bandId === filterBand
      return matchesSearch && matchesBand
    })
    .sort((a, b) => {
      switch (sortBy) {
        case "timestamp":
          return new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
        case "sar":
          return b.sarValue - a.sarValue
        case "gain":
          return b.gain - a.gain
        case "efficiency":
          return b.efficiency - a.efficiency
        default:
          return 0
      }
    })

  const uniqueBands = [...new Set(state.predictionHistory.map((p) => p.bandId))]

  return (
    <div className="space-y-6 p-6">
      <Card className="border-border/50 bg-gradient-to-br from-card to-muted/20">
        <CardHeader className="flex flex-row items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <History className="h-5 w-5 text-primary" />
              Prediction History
            </CardTitle>
            <p className="text-sm text-muted-foreground mt-1">View and manage your SAR prediction results</p>
          </div>
          {state.predictionHistory.length > 0 && (
            <Button onClick={handleExportAll} className="bg-gradient-to-r from-primary to-chart-2">
              <Download className="h-4 w-4 mr-2" />
              Export All
            </Button>
          )}
        </CardHeader>
        <CardContent>
          {state.predictionHistory.length === 0 ? (
            <div className="text-center py-12">
              <div className="h-20 w-20 mx-auto mb-4 rounded-full bg-gradient-to-br from-primary/20 to-chart-2/20 flex items-center justify-center">
                <History className="h-10 w-10 text-primary" />
              </div>
              <h3 className="text-lg font-semibold mb-2">No Predictions Yet</h3>
              <p className="text-muted-foreground mb-4">Run your first prediction to see results here</p>
              <Button
                onClick={() => dispatch({ type: "SET_ACTIVE_VIEW", payload: "dashboard" })}
                className="bg-gradient-to-r from-primary to-chart-2"
              >
                Start Predicting
              </Button>
            </div>
          ) : (
            <div className="space-y-4">
              {/* Filters and Search */}
              <div className="flex flex-col sm:flex-row gap-4">
                <div className="relative flex-1">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <Input
                    placeholder="Search predictions..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="pl-10 bg-background/50"
                  />
                </div>
                <Select value={filterBand} onValueChange={setFilterBand}>
                  <SelectTrigger className="w-full sm:w-48 bg-background/50">
                    <Filter className="h-4 w-4 mr-2" />
                    <SelectValue placeholder="Filter by band" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Bands</SelectItem>
                    {uniqueBands.map((band) => (
                      <SelectItem key={band} value={band}>
                        {band.toUpperCase()}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <Select value={sortBy} onValueChange={setSortBy}>
                  <SelectTrigger className="w-full sm:w-48 bg-background/50">
                    <SelectValue placeholder="Sort by" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="timestamp">Latest First</SelectItem>
                    <SelectItem value="sar">SAR Value</SelectItem>
                    <SelectItem value="gain">Gain</SelectItem>
                    <SelectItem value="efficiency">Efficiency</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Results Summary */}
              <div className="flex items-center justify-between text-sm text-muted-foreground">
                <span>
                  Showing {filteredPredictions.length} of {state.predictionHistory.length} predictions
                </span>
                <span>
                  Safe: {filteredPredictions.filter((p) => p.sarValue <= 1.6).length} | Warning:{" "}
                  {filteredPredictions.filter((p) => p.sarValue > 1.6 && p.sarValue <= 2.0).length} | Unsafe:{" "}
                  {filteredPredictions.filter((p) => p.sarValue > 2.0).length}
                </span>
              </div>

              {/* Table */}
              <div className="overflow-x-auto rounded-lg border border-border/50">
                <Table>
                  <TableHeader>
                    <TableRow className="bg-muted/50">
                      <TableHead className="font-semibold">Timestamp</TableHead>
                      <TableHead className="font-semibold">Band</TableHead>
                      <TableHead className="font-semibold">SAR (W/kg)</TableHead>
                      <TableHead className="font-semibold">Gain (dBi)</TableHead>
                      <TableHead className="font-semibold">Efficiency</TableHead>
                      <TableHead className="font-semibold">Safety</TableHead>
                      <TableHead className="font-semibold">Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {filteredPredictions.map((prediction) => (
                      <TableRow key={prediction.id} className="hover:bg-muted/30 transition-colors">
                        <TableCell className="font-mono text-xs">
                          {new Date(prediction.timestamp).toLocaleString()}
                        </TableCell>
                        <TableCell>
                          <Badge variant="outline" className="bg-gradient-to-r from-primary/10 to-chart-2/10">
                            {prediction.bandId.toUpperCase()}
                          </Badge>
                        </TableCell>
                        <TableCell className="font-mono font-semibold">{prediction.sarValue.toFixed(3)}</TableCell>
                        <TableCell>{prediction.gain.toFixed(1)}</TableCell>
                        <TableCell>{(prediction.efficiency * 100).toFixed(1)}%</TableCell>
                        <TableCell>{getSafetyBadge(prediction.sarValue)}</TableCell>
                        <TableCell>
                          <div className="flex gap-1">
                            <Button
                              size="sm"
                              variant="ghost"
                              onClick={() => handleViewPrediction(prediction)}
                              className="h-8 w-8 p-0 hover:bg-primary/10"
                            >
                              <Eye className="h-3 w-3" />
                            </Button>
                            <Button
                              size="sm"
                              variant="ghost"
                              onClick={() => handleDownloadPrediction(prediction)}
                              className="h-8 w-8 p-0 hover:bg-primary/10"
                            >
                              <Download className="h-3 w-3" />
                            </Button>
                            <Button
                              size="sm"
                              variant="ghost"
                              onClick={() => handleDeletePrediction(prediction.id)}
                              className="h-8 w-8 p-0 hover:bg-destructive/10 text-destructive"
                            >
                              <Trash2 className="h-3 w-3" />
                            </Button>
                          </div>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
