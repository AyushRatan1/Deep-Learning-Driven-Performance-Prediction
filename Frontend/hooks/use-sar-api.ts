"use client"

import { useEffect, useCallback } from "react"
import { useSAR } from "@/components/sar-context"
import { apiClient } from "@/lib/api-client"
import type { AntennaParameters, FrequencyBand, SARPrediction } from "@/components/sar-context"
import type { TrainingRequest, TrainingResponse, HealthcareApplications } from "@/lib/api-client"

export function useSARApi() {
  const { state, dispatch } = useSAR()

  // Load initial data on mount
  useEffect(() => {
    loadFrequencyBands()
    checkSystemStatus()
  }, [])

  const loadFrequencyBands = useCallback(async () => {
    try {
      const bands = await apiClient.getFrequencyBands()
      dispatch({ type: "SET_FREQUENCY_BANDS", payload: bands })
      
      // Set default band if none selected
      if (!state.selectedBand && bands.length > 0) {
        dispatch({ type: "SET_BAND", payload: bands[0] })
      }
    } catch (error) {
      console.error("Failed to load frequency bands:", error)
    }
  }, [dispatch, state.selectedBand])

  const checkSystemStatus = useCallback(async () => {
    try {
      const status = await apiClient.getSystemStatus()
      dispatch({ type: "SET_SYSTEM_STATUS", payload: status })
    } catch (error) {
      console.error("Failed to check system status:", error)
    }
  }, [dispatch])

  const generateSampleParameters = useCallback(async (bandId?: string) => {
    if (!bandId && !state.selectedBand) return
    
    const targetBandId = bandId || state.selectedBand!.id
    
    try {
      dispatch({ type: "SET_LOADING", payload: true })
      const parameters = await apiClient.generateSample(targetBandId)
      dispatch({ type: "SET_PARAMETERS", payload: parameters })
      dispatch({ type: "SET_INPUT_MODE", payload: "random" })
    } catch (error) {
      console.error("Failed to generate sample parameters:", error)
    } finally {
      dispatch({ type: "SET_LOADING", payload: false })
    }
  }, [state.selectedBand, dispatch])

  const predictSAR = useCallback(async (
    parameters?: AntennaParameters,
    bandId?: string
  ) => {
    const targetParameters = parameters || state.parameters
    const targetBandId = bandId || state.selectedBand?.id
    
    if (!targetBandId) {
      console.error("No frequency band selected")
      return
    }

    try {
      dispatch({ type: "SET_LOADING", payload: true })
      
      const prediction = await apiClient.predictSAR({
        band_id: targetBandId,
        parameters: targetParameters,
        input_mode: state.inputMode
      })

      dispatch({ type: "SET_PREDICTION", payload: prediction })
      dispatch({ type: "ADD_TO_HISTORY", payload: prediction })
      
      return prediction
    } catch (error) {
      console.error("Failed to predict SAR:", error)
      throw error
    } finally {
      dispatch({ type: "SET_LOADING", payload: false })
    }
  }, [state.parameters, state.selectedBand, state.inputMode, dispatch])

  const generateComparison = useCallback(async (
    parameters?: AntennaParameters,
    bandIds?: string[]
  ) => {
    const targetParameters = parameters || state.parameters
    const targetBandIds = bandIds || state.frequencyBands.map(b => b.id)
    
    try {
      dispatch({ type: "SET_LOADING", payload: true })
      
      const predictions: SARPrediction[] = []
      
      for (const bandId of targetBandIds) {
        try {
          const prediction = await apiClient.predictSAR({
            band_id: bandId,
            parameters: targetParameters,
            input_mode: state.inputMode
          })
          predictions.push(prediction)
        } catch (error) {
          console.error(`Failed to predict for band ${bandId}:`, error)
        }
      }
      
      dispatch({ type: "SET_COMPARISON_DATA", payload: predictions })
      return predictions
    } catch (error) {
      console.error("Failed to generate comparison:", error)
      throw error
    } finally {
      dispatch({ type: "SET_LOADING", payload: false })
    }
  }, [state.parameters, state.frequencyBands, state.inputMode, dispatch])

  const healthCheck = useCallback(async () => {
    try {
      const health = await apiClient.healthCheck()
      return health
    } catch (error) {
      console.error("Health check failed:", error)
      return { status: "error", timestamp: new Date().toISOString() }
    }
  }, [])

  const trainModel = useCallback(async (request: TrainingRequest): Promise<TrainingResponse | null> => {
    try {
      dispatch({ type: "SET_LOADING", payload: true })
      const response = await apiClient.trainModel(request)
      console.log("Model training successful:", response)
      return response
    } catch (error) {
      console.error("Training failed:", error)
      return null
    } finally {
      dispatch({ type: "SET_LOADING", payload: false })
    }
  }, [dispatch])

  const downloadDataset = useCallback(async (datasetType: string = "comprehensive") => {
    try {
      const response = await apiClient.downloadDataset(datasetType)
      console.log("Dataset download initiated:", response)
      return response
    } catch (error) {
      console.error("Dataset download failed:", error)
      return null
    }
  }, [])

  const getHealthcareApplications = useCallback(async (): Promise<HealthcareApplications | null> => {
    try {
      const response = await apiClient.getHealthcareApplications()
      console.log("Healthcare applications loaded:", response)
      return response
    } catch (error) {
      console.error("Failed to load healthcare applications:", error)
      return null
    }
  }, [])

  return {
    // API Methods
    generateSampleParameters,
    predictSAR,
    generateComparison,
    loadFrequencyBands,
    checkSystemStatus,
    healthCheck,
    trainModel,
    downloadDataset,
    getHealthcareApplications,
    
    // State
    isLoading: state.isLoading,
    selectedBand: state.selectedBand,
    frequencyBands: state.frequencyBands,
    currentPrediction: state.currentPrediction,
    predictionHistory: state.predictionHistory,
    comparisonData: state.comparisonData,
    systemStatus: state.systemStatus,
    parameters: state.parameters,
    inputMode: state.inputMode,
  }
}
