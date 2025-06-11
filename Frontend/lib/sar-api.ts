import type { AntennaParameters, SARPrediction, FrequencyBand } from "@/components/sar-context"
import { apiClient } from "./api-client"

// Enhanced SAR API service with real API integration
class SARAPIService {
  private fallbackToMock = true // Set to false when real API is available

  // Generate mock data for development/demo
  private generateMockSParameters(centerFreq: number, bandwidth = 2000) {
    const points = []
    const startFreq = centerFreq - bandwidth
    const endFreq = centerFreq + bandwidth
    const numPoints = 100

    for (let i = 0; i < numPoints; i++) {
      const freq = startFreq + (endFreq - startFreq) * (i / (numPoints - 1))
      const normalizedFreq = (freq - centerFreq) / bandwidth
      const s11 = -20 * Math.exp(-Math.pow(normalizedFreq * 3, 2)) - 5 + Math.random() * 2

      points.push({
        frequency: freq * 1000000, // Convert to Hz
        s11: Math.max(s11, -40),
      })
    }

    return points
  }

  private generateMockRadiationPattern() {
    const points = []
    const thetaSteps = 18
    const phiSteps = 36

    for (let i = 0; i < thetaSteps; i++) {
      for (let j = 0; j < phiSteps; j++) {
        const theta = (i * Math.PI) / (thetaSteps - 1)
        const phi = (j * 2 * Math.PI) / phiSteps

        const mainLobeGain = 15 * Math.cos(theta) * Math.cos(theta)
        const sideLobes = 5 * Math.sin(4 * phi) * Math.sin(2 * theta)
        const noise = (Math.random() - 0.5) * 2

        const gain = Math.max(mainLobeGain + sideLobes + noise, -20)
        points.push({ theta, phi, gain })
      }
    }

    return points
  }

  private delay(ms: number) {
    return new Promise((resolve) => setTimeout(resolve, ms))
  }

  async getBands(): Promise<FrequencyBand[]> {
    try {
      if (!this.fallbackToMock) {
        const response = await apiClient.getFrequencyBands()
        if (response.success && response.data) {
          return response.data
        }
      }
    } catch (error) {
      console.warn("Failed to fetch from real API, falling back to mock data:", error)
    }

    // Fallback to mock data
    await this.delay(200)
    return [
      { id: "x-band", name: "X-band", range: "8-12 GHz", frequency: 10, color: "#3b82f6" },
      { id: "ku-band", name: "Ku-band", range: "12-18 GHz", frequency: 15, color: "#8b5cf6" },
      { id: "k-band", name: "K-band", range: "18-27 GHz", frequency: 22.5, color: "#06b6d4" },
      { id: "ka-band", name: "Ka-band", range: "27-40 GHz", frequency: 33.5, color: "#10b981" },
      { id: "v-band", name: "V-band", range: "40-75 GHz", frequency: 57.5, color: "#f59e0b" },
      { id: "w-band", name: "W-band", range: "75-110 GHz", frequency: 92.5, color: "#ef4444" },
    ]
  }

  async predict(bandId: string, parameters: AntennaParameters): Promise<SARPrediction> {
    try {
      if (!this.fallbackToMock) {
        const response = await apiClient.createPrediction({
          frequencyBandId: bandId,
          parameters,
          userId: "demo-user", // In real app, get from auth context
        })

        if (response.success && response.data) {
          return response.data
        }
      }
    } catch (error) {
      console.warn("Failed to create prediction via API, falling back to mock:", error)
    }

    // Fallback to mock prediction
    await this.delay(1500)

    const bands = await this.getBands()
    const band = bands.find((b) => b.id === bandId)
    const centerFreq = band?.frequency || 10

    // Simulate SAR calculation
    const baseSAR = 0.8 + Math.random() * 1.5
    const frequencyFactor = Math.log(centerFreq / 10) * 0.3
    const sizeFactor = ((parameters.patchWidth * parameters.patchLength) / 150) * 0.2
    const substrateFactor = (parameters.substrateThickness / 2) * 0.1

    const sarValue = Math.max(0.1, baseSAR + frequencyFactor + sizeFactor + substrateFactor)
    const gain = 5 + Math.random() * 10 + Math.log(centerFreq / 10) * 2
    const efficiency = 0.6 + Math.random() * 0.35
    const bandwidth = 200 + Math.random() * 800 + centerFreq * 20

    const prediction: SARPrediction = {
      id: `pred_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date().toISOString(),
      bandId,
      bandName: band?.name || bandId,
      parameters,
      sarValue,
      gain,
      efficiency,
      bandwidth,
      sParameters: this.generateMockSParameters(centerFreq),
      radiationPattern: this.generateMockRadiationPattern(),
    }

    return prediction
  }

  async getPredictions(filters: any = {}): Promise<SARPrediction[]> {
    try {
      if (!this.fallbackToMock) {
        const response = await apiClient.getPredictions(filters)
        if (response.success && response.data) {
          return response.data.predictions
        }
      }
    } catch (error) {
      console.warn("Failed to fetch predictions from API:", error)
    }

    // Return empty array for mock
    await this.delay(300)
    return []
  }

  async deletePrediction(predictionId: string): Promise<boolean> {
    try {
      if (!this.fallbackToMock) {
        const response = await apiClient.deletePrediction(predictionId)
        return response.success
      }
    } catch (error) {
      console.warn("Failed to delete prediction via API:", error)
    }

    await this.delay(500)
    return true
  }

  async getComparisonData(): Promise<SARPrediction[]> {
    try {
      if (!this.fallbackToMock) {
        const bands = await this.getBands()
        const defaultParams = {
          substrateThickness: 1.6,
          permittivity: 4.4,
          patchWidth: 10.0,
          patchLength: 12.0,
          feedPosition: 0.25,
        }

        const response = await apiClient.generateComparison({
          parameters: defaultParams,
          frequencyBands: bands.map((b) => b.id),
        })

        if (response.success && response.data) {
          return response.data
        }
      }
    } catch (error) {
      console.warn("Failed to generate comparison via API:", error)
    }

    // Fallback to mock comparison
    await this.delay(1000)
    const bands = ["x-band", "ku-band", "k-band", "ka-band", "v-band", "w-band"]
    const predictions = []

    for (const bandId of bands) {
      const defaultParams = {
        substrateThickness: 1.6,
        permittivity: 4.4,
        patchWidth: 10.0,
        patchLength: 12.0,
        feedPosition: 0.25,
      }
      const prediction = await this.predict(bandId, defaultParams)
      predictions.push(prediction)
    }

    return predictions
  }

  async exportData(format: "csv" | "json" | "xlsx", data: any[]): Promise<string> {
    try {
      if (!this.fallbackToMock) {
        const response = await apiClient.exportPredictions(format, {})
        if (response.success && response.data) {
          return response.data.downloadUrl
        }
      }
    } catch (error) {
      console.warn("Failed to export via API:", error)
    }

    // Mock export - generate download URL
    await this.delay(1000)
    return `https://api.sarpredictor.com/downloads/export_${Date.now()}.${format}`
  }

  async getAnalytics(timeRange = "30d") {
    try {
      if (!this.fallbackToMock) {
        const response = await apiClient.getAnalytics(timeRange)
        if (response.success && response.data) {
          return response.data
        }
      }
    } catch (error) {
      console.warn("Failed to fetch analytics from API:", error)
    }

    // Mock analytics data
    await this.delay(500)
    return {
      totalPredictions: 1247,
      safetyDistribution: { safe: 892, warning: 234, unsafe: 121 },
      frequencyBandUsage: [
        { band: "X-band", count: 345 },
        { band: "Ku-band", count: 298 },
        { band: "K-band", count: 234 },
        { band: "Ka-band", count: 189 },
        { band: "V-band", count: 123 },
        { band: "W-band", count: 58 },
      ],
      averageSAR: 1.234,
      trendsData: Array.from({ length: 30 }, (_, i) => ({
        date: new Date(Date.now() - (29 - i) * 24 * 60 * 60 * 1000).toISOString().split("T")[0],
        predictions: Math.floor(Math.random() * 50) + 10,
        averageSAR: 1.0 + Math.random() * 0.8,
      })),
    }
  }
}

export const sarAPI = new SARAPIService()
