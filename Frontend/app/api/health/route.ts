import { NextResponse } from "next/server"

export async function GET() {
  try {
    // In a real application, you would check database connectivity,
    // external service health, etc.

    const healthData = {
      status: "healthy",
      timestamp: new Date().toISOString(),
      version: "2.1.0",
      uptime: process.uptime(),
      environment: process.env.NODE_ENV || "development",
      services: {
        database: "connected",
        redis: "connected",
        ml_service: "connected",
      },
    }

    return NextResponse.json({
      success: true,
      data: healthData,
    })
  } catch (error) {
    return NextResponse.json(
      {
        success: false,
        error: "Health check failed",
        message: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 },
    )
  }
}
