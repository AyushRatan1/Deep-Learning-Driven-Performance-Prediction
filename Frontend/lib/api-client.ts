// Real API client for SAR Predictor
import type { AntennaParameters, SARPrediction, FrequencyBand } from "@/components/sar-context"

interface APIResponse<T> {
  success: boolean
  data?: T
  error?: string
  message?: string
}

interface PredictionRequest {
  band_id: string
  parameters: AntennaParameters
  input_mode: string
}

interface ChatRequest {
  message: string
  context?: any
}

interface ChatResponse {
  response: string
  timestamp: string
  context?: any
}

interface TrainingRequest {
  num_samples: number
  include_real_data: boolean
  focus_applications: string[]
}

interface TrainingResponse {
  status: string
  message: string
  samples_generated: number
  training_time: number
  model_accuracy: number
  timestamp: string
}

interface HealthcareApplications {
  total_categories: number
  applications: Record<string, any>
  regulatory_info: Record<string, string>
}

class SARAPIClient {
  private baseURL: string

  constructor() {
    this.baseURL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000"
  }

  private async request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const url = `${this.baseURL}${endpoint}`

    const config: RequestInit = {
      headers: {
        "Content-Type": "application/json",
        ...options.headers,
      },
      ...options,
    }

    try {
      const response = await fetch(url, config)
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      return data
    } catch (error) {
      console.error(`API request failed: ${error}`)
      throw error
    }
  }

  async getFrequencyBands(): Promise<FrequencyBand[]> {
    try {
      const response = await this.request<{success: boolean, data: FrequencyBand[], count: number, fallback?: boolean}>("/bands")
      return response.data || []
    } catch (error) {
      console.error("Failed to load frequency bands:", error)
      return []
    }
  }

  async predictSAR(request: PredictionRequest): Promise<SARPrediction> {
    const response = await this.request<{success: boolean, data: SARPrediction}>("/predict", {
      method: "POST",
      body: JSON.stringify(request),
    })
    return response.data
  }

  async generateSample(bandId: string): Promise<AntennaParameters> {
    const response = await this.request<{ parameters: AntennaParameters }>(`/generate-sample/${bandId}`)
    return response.parameters
  }

  async chatWithAI(request: ChatRequest): Promise<ChatResponse> {
    return this.request<ChatResponse>("/chat", {
      method: "POST",
      body: JSON.stringify(request),
    })
  }

  async getSystemStatus(): Promise<any> {
    try {
      return this.request<any>("/system-status")
    } catch (error) {
      console.error("System status check failed:", error)
      return { status: "offline", error: "Failed to connect to backend" }
    }
  }

  async healthCheck(): Promise<{ status: string; timestamp: string }> {
    try {
      return this.request<{ status: string; timestamp: string }>("/health")
    } catch (error) {
      console.error("Health check failed:", error)
      return { status: "offline", timestamp: new Date().toISOString() }
    }
  }

  async trainModel(request: TrainingRequest): Promise<TrainingResponse> {
    return this.request<TrainingResponse>("/train-model", {
      method: "POST",
      body: JSON.stringify(request),
    })
  }

  async downloadDataset(datasetType: string = "comprehensive"): Promise<any> {
    return this.request<any>(`/download-dataset/${datasetType}`)
  }

  async getDatasetInfo(): Promise<any> {
    return this.request<any>("/dataset-info")
  }

  async getHealthcareApplications(): Promise<HealthcareApplications> {
    return this.request<HealthcareApplications>("/healthcare-applications")
  }
}

export const apiClient = new SARAPIClient()
export type { PredictionRequest, ChatRequest, ChatResponse, TrainingRequest, TrainingResponse, HealthcareApplications }
