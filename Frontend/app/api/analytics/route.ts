import { type NextRequest, NextResponse } from "next/server"

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const timeRange = searchParams.get("timeRange") || "30d"

    // Generate mock analytics data
    const analytics = {
      totalPredictions: 1247,
      safetyDistribution: {
        safe: 892,
        warning: 234,
        unsafe: 121,
      },
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

    return NextResponse.json({
      success: true,
      data: analytics,
    })
  } catch (error) {
    return NextResponse.json(
      {
        success: false,
        error: "Failed to fetch analytics",
        message: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 },
    )
  }
}
