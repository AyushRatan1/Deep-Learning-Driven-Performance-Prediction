"use client"

import { useState, useEffect } from "react"
import { useSAR } from "./sar-context"
import { useSARApi } from "@/hooks/use-sar-api"
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Separator } from "@/components/ui/separator"
import {
  BarChart3,
  Radio,
  MessageSquare,
  Settings,
  Zap,
  Activity,
  TrendingUp,
  Database,
  Wifi,
  Brain,
  Shield,
  Target,
  Clock,
  MapPin,
} from "lucide-react"

export function AppSidebar() {
  const { state, dispatch } = useSAR()
  const { 
    isLoading, 
    currentPrediction, 
    predictionHistory, 
    selectedBand, 
    frequencyBands,
    systemStatus 
  } = useSARApi()

  // Calculate quick stats
  const totalPredictions = predictionHistory.length
  const avgSAR = predictionHistory.length > 0 
    ? predictionHistory.reduce((sum, p) => sum + p.sar_value, 0) / predictionHistory.length 
    : 0
  const latestPrediction = predictionHistory[0]

  // Navigation items
  const navItems = [
    {
      title: "Overview",
      icon: BarChart3,
      view: "dashboard",
      description: "System dashboard and metrics",
      badge: totalPredictions > 0 ? String(totalPredictions) : null,
    },
    {
      title: "SAR Prediction",
      icon: Zap,
      view: "comparison",
      description: "Run antenna predictions",
      badge: isLoading ? "Running" : null,
      badgeVariant: isLoading ? "default" : "outline",
    },
    {
      title: "Professional Map",
      icon: MapPin,
      view: "professional-map",
      description: "Real-world SAR coverage mapping",
      badge: "Mapbox",
      badgeVariant: "secondary",
    },
    {
      title: "History",
      icon: Database,
      view: "history", 
      description: "View past predictions",
      badge: predictionHistory.length > 5 ? "5+" : predictionHistory.length > 0 ? String(predictionHistory.length) : null,
    },
    {
      title: "AI Assistant",
      icon: MessageSquare,
      view: "chat",
      description: "Get antenna design help",
      badge: "AI",
      badgeVariant: "secondary",
    },
    {
      title: "Settings",
      icon: Settings,
      view: "settings",
      description: "System configuration",
      badge: systemStatus?.status === "healthy" ? "Online" : "Offline",
      badgeVariant: systemStatus?.status === "healthy" ? "default" : "destructive",
    },
  ]

  return (
    <Sidebar className="border-r-2 border-border/40 bg-card/30 backdrop-blur-sm">
      <SidebarHeader className="p-6 border-b border-border/40">
        <div className="flex items-center space-x-3">
          <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-primary to-primary/60 flex items-center justify-center">
            <Radio className="w-7 h-7 text-white" />
          </div>
          <div className="space-y-1">
            <h1 className="text-xl font-bold tracking-tight">SAR Predictor</h1>
            <p className="text-sm text-muted-foreground">Antenna Analysis Pro</p>
          </div>
        </div>
        
        {/* Quick system status */}
        <div className="mt-4 p-3 rounded-lg bg-secondary/50 border">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium">System Status</span>
            <div className="flex items-center space-x-1">
              <div className={`w-2 h-2 rounded-full ${systemStatus?.status === "healthy" ? "bg-green-500" : "bg-yellow-500"}`} />
              <span className="text-xs text-muted-foreground">
                {systemStatus?.status === "healthy" ? "Online" : "Checking..."}
              </span>
            </div>
          </div>
          <Progress 
            value={systemStatus?.status === "healthy" ? 100 : 50} 
            className="h-2"
          />
        </div>
      </SidebarHeader>

      <SidebarContent className="px-4">
        <SidebarGroup>
          <SidebarGroupLabel className="text-base font-semibold px-2 py-3 text-foreground">
            Navigation
          </SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {navItems.map((item) => (
                <SidebarMenuItem key={item.view}>
                  <SidebarMenuButton
                    onClick={() => dispatch({ type: "SET_ACTIVE_VIEW", payload: item.view as any })}
                    isActive={state.activeView === item.view}
                    className="w-full h-14 px-4 text-left justify-start hover:bg-secondary/60 data-[active=true]:bg-primary/10 data-[active=true]:text-primary data-[active=true]:border-primary/20 border border-transparent transition-all duration-200"
                  >
                    <div className="flex items-center space-x-3 w-full">
                      <item.icon className="w-5 h-5 flex-shrink-0" />
                      <div className="flex-1 min-w-0">
                        <div className="text-sm font-medium truncate">{item.title}</div>
                        <div className="text-xs text-muted-foreground truncate">{item.description}</div>
                      </div>
                      {item.badge && (
                        <Badge variant={item.badgeVariant as any || "outline"} className="text-xs px-2 py-0.5 flex-shrink-0">
                          {item.badge}
                        </Badge>
                      )}
                    </div>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        <Separator className="my-4 mx-2" />

        {/* Current Band Selection */}
        <SidebarGroup>
          <SidebarGroupLabel className="text-base font-semibold px-2 py-3 text-foreground">
            Active Band
          </SidebarGroupLabel>
          <SidebarGroupContent>
            <div className="px-2 space-y-3">
              {selectedBand ? (
                <div className="p-4 rounded-lg bg-secondary/30 border border-border/40">
                  <div className="flex items-center space-x-2 mb-2">
                    <Target className="w-4 h-4 text-primary" />
                    <span className="text-sm font-medium">{selectedBand.name}</span>
                  </div>
                  <div className="space-y-1 text-xs text-muted-foreground">
                    <div>Range: {selectedBand.range}</div>
                    <div>Center: {selectedBand.center_freq} GHz</div>
                  </div>
                  <Badge 
                    variant="outline" 
                    className="mt-2 text-xs"
                    style={{ borderColor: selectedBand.color, color: selectedBand.color }}
                  >
                    Active
                  </Badge>
                </div>
              ) : (
                <div className="p-4 rounded-lg border border-dashed border-border/40 text-center">
                  <Radio className="w-8 h-8 mx-auto text-muted-foreground mb-2" />
                  <p className="text-sm text-muted-foreground">No band selected</p>
                  <p className="text-xs text-muted-foreground">Choose from Overview tab</p>
                </div>
              )}
            </div>
          </SidebarGroupContent>
        </SidebarGroup>

        <Separator className="my-4 mx-2" />

        {/* Quick Stats */}
        <SidebarGroup>
          <SidebarGroupLabel className="text-base font-semibold px-2 py-3 text-foreground">
            Quick Stats
          </SidebarGroupLabel>
          <SidebarGroupContent>
            <div className="px-2 space-y-3">
              <div className="grid grid-cols-1 gap-3">
                <div className="p-3 rounded-lg bg-secondary/30 border border-border/40">
                  <div className="flex items-center space-x-2 mb-1">
                    <Database className="w-4 h-4 text-blue-500" />
                    <span className="text-sm font-medium">Predictions</span>
                  </div>
                  <div className="text-xl font-bold">{totalPredictions}</div>
                  <div className="text-xs text-muted-foreground">Total runs</div>
                </div>

                {latestPrediction && (
                  <div className="p-3 rounded-lg bg-secondary/30 border border-border/40">
                    <div className="flex items-center space-x-2 mb-1">
                      <Zap className="w-4 h-4 text-orange-500" />
                      <span className="text-sm font-medium">Latest SAR</span>
                    </div>
                    <div className="text-xl font-bold">{(latestPrediction?.sar_value || 0).toFixed(3)}</div>
                    <div className="text-xs text-muted-foreground">W/kg</div>
                  </div>
                )}

                {avgSAR > 0 && (
                  <div className="p-3 rounded-lg bg-secondary/30 border border-border/40">
                    <div className="flex items-center space-x-2 mb-1">
                      <TrendingUp className="w-4 h-4 text-green-500" />
                      <span className="text-sm font-medium">Avg SAR</span>
                    </div>
                    <div className="text-xl font-bold">{(avgSAR || 0).toFixed(3)}</div>
                    <div className="text-xs text-muted-foreground">W/kg</div>
                  </div>
                )}
              </div>
            </div>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>

      <SidebarFooter className="p-4 border-t border-border/40">
        <div className="space-y-3">
          {/* Latest Prediction Summary */}
          {currentPrediction && (
            <div className="p-3 rounded-lg bg-primary/5 border border-primary/20">
              <div className="flex items-center space-x-2 mb-2">
                <Shield className="w-4 h-4 text-primary" />
                <span className="text-sm font-medium text-primary">Latest Result</span>
              </div>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div>
                  <div className="text-muted-foreground">SAR</div>
                  <div className="font-medium">{(currentPrediction?.sar_value || 0).toFixed(3)} W/kg</div>
                </div>
                <div>
                  <div className="text-muted-foreground">Gain</div>
                  <div className="font-medium">{(currentPrediction?.gain || 0).toFixed(1)} dBi</div>
                </div>
              </div>
            </div>
          )}

          {/* Model Info */}
          <div className="flex items-center space-x-2 text-xs text-muted-foreground">
            <Brain className="w-4 h-4" />
            <span>Physics-based SAR Model v2.0</span>
          </div>
          
          {/* Performance indicator */}
          <div className="flex items-center justify-between text-xs">
            <div className="flex items-center space-x-1">
              <Activity className="w-3 h-3 text-green-500" />
              <span className="text-muted-foreground">Performance</span>
            </div>
            <Badge variant="outline" className="text-xs px-2 py-0.5">
              Excellent
            </Badge>
          </div>
        </div>
      </SidebarFooter>
    </Sidebar>
  )
}
