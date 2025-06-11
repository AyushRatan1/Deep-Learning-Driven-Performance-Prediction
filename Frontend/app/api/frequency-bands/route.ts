import { NextResponse } from "next/server"

const frequencyBands = [
  { id: "x-band", name: "X-band", range: "8-12 GHz", frequency: 10, color: "#3b82f6" },
  { id: "ku-band", name: "Ku-band", range: "12-18 GHz", frequency: 15, color: "#8b5cf6" },
  { id: "k-band", name: "K-band", range: "18-27 GHz", frequency: 22.5, color: "#06b6d4" },
  { id: "ka-band", name: "Ka-band", range: "27-40 GHz", frequency: 33.5, color: "#10b981" },
  { id: "v-band", name: "V-band", range: "40-75 GHz", frequency: 57.5, color: "#f59e0b" },
  { id: "w-band", name: "W-band", range: "75-110 GHz", frequency: 92.5, color: "#ef4444" },
  { id: "d-band", name: "D-band", range: "110-170 GHz", frequency: 140, color: "#ec4899" },
]

export async function GET() {
  try {
    return NextResponse.json({
      success: true,
      data: frequencyBands,
    })
  } catch (error) {
    return NextResponse.json(
      {
        success: false,
        error: "Failed to fetch frequency bands",
        message: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 },
    )
  }
}
