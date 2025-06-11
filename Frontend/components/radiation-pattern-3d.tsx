"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Canvas } from "@react-three/fiber"
import { OrbitControls, Environment, Html } from "@react-three/drei"
import { useSAR } from "@/components/sar-context"
import { useMemo, useState } from "react"
import { Download, Maximize2, RotateCcw } from "lucide-react"
import * as THREE from "three"

function RadiationPatternMesh({ data }: { data: Array<{ theta: number; phi: number; gain: number }> }) {
  const geometry = useMemo(() => {
    const geometry = new THREE.SphereGeometry(1, 32, 16)
    const positions = geometry.attributes.position.array as Float32Array

    for (let i = 0; i < positions.length; i += 3) {
      const x = positions[i]
      const y = positions[i + 1]
      const z = positions[i + 2]

      const r = Math.sqrt(x * x + y * y + z * z)
      const theta = Math.acos(z / r)
      const phi = Math.atan2(y, x)

      let minDist = Number.POSITIVE_INFINITY
      let closestGain = 0

      data.forEach((point) => {
        const dist = Math.abs(theta - point.theta) + Math.abs(phi - point.phi)
        if (dist < minDist) {
          minDist = dist
          closestGain = point.gain
        }
      })

      const normalizedGain = Math.max(0, (closestGain + 20) / 40)
      const scaleFactor = 0.3 + normalizedGain * 0.7

      positions[i] = x * scaleFactor
      positions[i + 1] = y * scaleFactor
      positions[i + 2] = z * scaleFactor
    }

    geometry.attributes.position.needsUpdate = true
    geometry.computeVertexNormals()

    return geometry
  }, [data])

  return (
    <mesh geometry={geometry}>
      <meshStandardMaterial
        color="#3b82f6"
        transparent
        opacity={0.8}
        wireframe={false}
        roughness={0.3}
        metalness={0.7}
      />
    </mesh>
  )
}

export function RadiationPattern3D() {
  const { state } = useSAR()
  const { currentPrediction } = state
  const [isRotating, setIsRotating] = useState(true)

  const handleDownload = () => {
    if (!currentPrediction) return

    const csvContent =
      "data:text/csv;charset=utf-8," +
      "Theta (rad),Phi (rad),Gain (dB)\n" +
      currentPrediction.radiationPattern
        .map((point) => `${point.theta.toFixed(6)},${point.phi.toFixed(6)},${point.gain.toFixed(3)}`)
        .join("\n")

    const encodedUri = encodeURI(csvContent)
    const link = document.createElement("a")
    link.setAttribute("href", encodedUri)
    link.setAttribute("download", `radiation_pattern_${currentPrediction.id}.csv`)
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  const handleMaximize = () => {
    console.log("Maximize 3D view")
  }

  const toggleRotation = () => {
    setIsRotating(!isRotating)
  }

  if (!currentPrediction || !currentPrediction.radiationPattern) {
    return (
      <Card className="border-border/50 bg-gradient-to-br from-card to-muted/20">
        <CardHeader>
          <CardTitle className="text-sm font-medium">3D Radiation Pattern</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-64 flex items-center justify-center text-muted-foreground">
            <div className="text-center">
              <div className="h-16 w-16 mx-auto mb-3 rounded-full bg-muted/50 flex items-center justify-center">🌐</div>
              <p>No radiation pattern data available</p>
            </div>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className="border-border/50 bg-gradient-to-br from-card to-muted/20">
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="text-sm font-medium">3D Radiation Pattern</CardTitle>
        <div className="flex gap-2">
          <Button size="sm" variant="ghost" onClick={toggleRotation}>
            <RotateCcw className={`h-4 w-4 ${isRotating ? "animate-spin" : ""}`} />
          </Button>
          <Button size="sm" variant="ghost" onClick={handleDownload}>
            <Download className="h-4 w-4" />
          </Button>
          <Button size="sm" variant="ghost" onClick={handleMaximize}>
            <Maximize2 className="h-4 w-4" />
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <div className="h-64 rounded-lg overflow-hidden bg-gradient-to-br from-slate-900 to-slate-800">
          <Canvas camera={{ position: [3, 3, 3], fov: 50 }}>
            <Environment preset="studio" />
            <ambientLight intensity={0.5} />
            <pointLight position={[10, 10, 10]} intensity={1} />
            <pointLight position={[-10, -10, -10]} intensity={0.5} />
            <RadiationPatternMesh data={currentPrediction.radiationPattern} />
            <OrbitControls
              enablePan={true}
              enableZoom={true}
              enableRotate={true}
              autoRotate={isRotating}
              autoRotateSpeed={2}
            />
            <Html position={[0, -1.5, 0]} center>
              <div className="text-xs text-white bg-black/50 px-2 py-1 rounded">
                Peak Gain: {Math.max(...currentPrediction.radiationPattern.map((p) => p.gain)).toFixed(1)} dBi
              </div>
            </Html>
          </Canvas>
        </div>
        <div className="text-xs text-muted-foreground mt-2 text-center">
          Drag to rotate • Scroll to zoom • Right-click to pan • Auto-rotation: {isRotating ? "ON" : "OFF"}
        </div>
      </CardContent>
    </Card>
  )
}
