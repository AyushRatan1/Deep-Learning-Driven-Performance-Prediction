"use client"

import type React from "react"
import { createContext, useContext, useReducer, type ReactNode } from "react"

export interface FrequencyBand {
  id: string
  name: string
  range: string
  min_freq: number
  max_freq: number
  center_freq: number
  color: string
}

export interface AntennaParameters {
  substrate_thickness: number
  substrate_permittivity: number
  patch_width: number
  patch_length: number
  bending_radius: number
  power_density: number
}

export interface SARPrediction {
  id: string
  timestamp: string
  band_id: string
  band_name: string
  parameters: AntennaParameters
  sar_value: number
  gain: number
  efficiency: number
  bandwidth: number
  frequency: number
  s_parameters: Array<{ frequency: number; s11: number }>
  radiation_pattern: Array<{ theta: number; phi: number; gain: number }>
  max_return_loss: number
  resonant_frequency: number
}

export interface ChatMessage {
  id: string
  message: string
  response: string
  timestamp: string
  isUser: boolean
}

interface SARState {
  selectedBand: FrequencyBand | null
  inputMode: "random" | "custom"
  parameters: AntennaParameters
  currentPrediction: SARPrediction | null
  predictionHistory: SARPrediction[]
  isLoading: boolean
  comparisonData: SARPrediction[]
  activeView: "dashboard" | "comparison" | "history" | "settings" | "chat"
  selectedPrediction: SARPrediction | null
  chatMessages: ChatMessage[]
  frequencyBands: FrequencyBand[]
  systemStatus: any
}

type SARAction =
  | { type: "SET_BAND"; payload: FrequencyBand }
  | { type: "SET_INPUT_MODE"; payload: "random" | "custom" }
  | { type: "SET_PARAMETERS"; payload: Partial<AntennaParameters> }
  | { type: "SET_PREDICTION"; payload: SARPrediction }
  | { type: "ADD_TO_HISTORY"; payload: SARPrediction }
  | { type: "SET_LOADING"; payload: boolean }
  | { type: "SET_COMPARISON_DATA"; payload: SARPrediction[] }
  | { type: "SET_ACTIVE_VIEW"; payload: "dashboard" | "comparison" | "history" | "settings" | "chat" }
  | { type: "SET_SELECTED_PREDICTION"; payload: SARPrediction | null }
  | { type: "ADD_CHAT_MESSAGE"; payload: ChatMessage }
  | { type: "SET_FREQUENCY_BANDS"; payload: FrequencyBand[] }
  | { type: "SET_SYSTEM_STATUS"; payload: any }

const initialState: SARState = {
  selectedBand: null,
  inputMode: "random",
  parameters: {
    substrate_thickness: 1.6,
    substrate_permittivity: 4.4,
    patch_width: 10.0,
    patch_length: 12.0,
    bending_radius: 50.0,
    power_density: 1.0,
  },
  currentPrediction: null,
  predictionHistory: [],
  isLoading: false,
  comparisonData: [],
  activeView: "dashboard",
  selectedPrediction: null,
  chatMessages: [],
  frequencyBands: [],
  systemStatus: null,
}

function sarReducer(state: SARState, action: SARAction): SARState {
  switch (action.type) {
    case "SET_BAND":
      return { ...state, selectedBand: action.payload }
    case "SET_INPUT_MODE":
      return { ...state, inputMode: action.payload }
    case "SET_PARAMETERS":
      return { ...state, parameters: { ...state.parameters, ...action.payload } }
    case "SET_PREDICTION":
      return { ...state, currentPrediction: action.payload }
    case "ADD_TO_HISTORY":
      return { ...state, predictionHistory: [action.payload, ...state.predictionHistory] }
    case "SET_LOADING":
      return { ...state, isLoading: action.payload }
    case "SET_COMPARISON_DATA":
      return { ...state, comparisonData: action.payload }
    case "SET_ACTIVE_VIEW":
      return { ...state, activeView: action.payload }
    case "SET_SELECTED_PREDICTION":
      return { ...state, selectedPrediction: action.payload }
    case "ADD_CHAT_MESSAGE":
      return { ...state, chatMessages: [...state.chatMessages, action.payload] }
    case "SET_FREQUENCY_BANDS":
      return { ...state, frequencyBands: action.payload }
    case "SET_SYSTEM_STATUS":
      return { ...state, systemStatus: action.payload }
    default:
      return state
  }
}

const SARContext = createContext<{
  state: SARState
  dispatch: React.Dispatch<SARAction>
} | null>(null)

export function SARProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(sarReducer, initialState)

  return <SARContext.Provider value={{ state, dispatch }}>{children}</SARContext.Provider>
}

export function useSAR() {
  const context = useContext(SARContext)
  if (!context) {
    throw new Error("useSAR must be used within a SARProvider")
  }
  return context
}
