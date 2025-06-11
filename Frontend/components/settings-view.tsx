"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"
import { Separator } from "@/components/ui/separator"
import { Settings, Palette, Database, Bell, Shield, Download } from "lucide-react"
import { useTheme } from "next-themes"
import { useState } from "react"

export function SettingsView() {
  const { theme, setTheme } = useTheme()
  const [notifications, setNotifications] = useState(true)
  const [autoSave, setAutoSave] = useState(true)
  const [dataRetention, setDataRetention] = useState(30)
  const [exportFormat, setExportFormat] = useState("csv")

  const handleExportSettings = () => {
    const settings = {
      theme,
      notifications,
      autoSave,
      dataRetention,
      exportFormat,
      exportedAt: new Date().toISOString(),
    }

    const jsonContent = "data:text/json;charset=utf-8," + JSON.stringify(settings, null, 2)
    const encodedUri = encodeURI(jsonContent)
    const link = document.createElement("a")
    link.setAttribute("href", encodedUri)
    link.setAttribute("download", "sar_predictor_settings.json")
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  const handleResetSettings = () => {
    setTheme("system")
    setNotifications(true)
    setAutoSave(true)
    setDataRetention(30)
    setExportFormat("csv")
  }

  return (
    <div className="space-y-6 p-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold flex items-center gap-2">
            <Settings className="h-6 w-6 text-primary" />
            Settings & Configuration
          </h2>
          <p className="text-muted-foreground">Customize your SAR Predictor Pro experience</p>
        </div>
        <div className="flex gap-2">
          <Button onClick={handleExportSettings} variant="outline">
            <Download className="h-4 w-4 mr-2" />
            Export Settings
          </Button>
          <Button onClick={handleResetSettings} variant="outline">
            Reset to Defaults
          </Button>
        </div>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        {/* Appearance Settings */}
        <Card className="border-border/50 bg-gradient-to-br from-card to-muted/20">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Palette className="h-5 w-5 text-primary" />
              Appearance
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label>Theme</Label>
              <Select value={theme} onValueChange={setTheme}>
                <SelectTrigger>
                  <SelectValue placeholder="Select theme" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="light">Light</SelectItem>
                  <SelectItem value="dark">Dark</SelectItem>
                  <SelectItem value="system">System</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <Separator />

            <div className="space-y-4">
              <h4 className="font-medium">Color Scheme Preview</h4>
              <div className="grid grid-cols-4 gap-2">
                <div className="h-8 rounded bg-primary"></div>
                <div className="h-8 rounded bg-chart-1"></div>
                <div className="h-8 rounded bg-chart-2"></div>
                <div className="h-8 rounded bg-chart-3"></div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Notifications */}
        <Card className="border-border/50 bg-gradient-to-br from-card to-muted/20">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Bell className="h-5 w-5 text-primary" />
              Notifications
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="space-y-0.5">
                <Label>Enable Notifications</Label>
                <p className="text-sm text-muted-foreground">Get notified when predictions complete</p>
              </div>
              <Switch checked={notifications} onCheckedChange={setNotifications} />
            </div>

            <Separator />

            <div className="space-y-2">
              <Label>Notification Types</Label>
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm">Prediction Complete</span>
                  <Switch checked={notifications} disabled={!notifications} />
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm">Safety Warnings</span>
                  <Switch checked={notifications} disabled={!notifications} />
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm">System Updates</span>
                  <Switch checked={false} disabled={!notifications} />
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Data Management */}
        <Card className="border-border/50 bg-gradient-to-br from-card to-muted/20">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Database className="h-5 w-5 text-primary" />
              Data Management
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="space-y-0.5">
                <Label>Auto-save Predictions</Label>
                <p className="text-sm text-muted-foreground">Automatically save prediction results</p>
              </div>
              <Switch checked={autoSave} onCheckedChange={setAutoSave} />
            </div>

            <Separator />

            <div className="space-y-2">
              <Label>Data Retention: {dataRetention} days</Label>
              <Slider
                value={[dataRetention]}
                onValueChange={(value) => setDataRetention(value[0])}
                max={365}
                min={7}
                step={1}
                className="w-full"
              />
              <p className="text-xs text-muted-foreground">Predictions older than this will be automatically deleted</p>
            </div>

            <Separator />

            <div className="space-y-2">
              <Label>Default Export Format</Label>
              <Select value={exportFormat} onValueChange={setExportFormat}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="csv">CSV</SelectItem>
                  <SelectItem value="json">JSON</SelectItem>
                  <SelectItem value="xlsx">Excel</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </CardContent>
        </Card>

        {/* Safety & Compliance */}
        <Card className="border-border/50 bg-gradient-to-br from-card to-muted/20">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Shield className="h-5 w-5 text-primary" />
              Safety & Compliance
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label>SAR Limit Standard</Label>
              <Select defaultValue="fcc">
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="fcc">FCC (1.6 W/kg)</SelectItem>
                  <SelectItem value="icnirp">ICNIRP (2.0 W/kg)</SelectItem>
                  <SelectItem value="custom">Custom Limit</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <Separator />

            <div className="space-y-2">
              <Label>Safety Warnings</Label>
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm">Show warning at 80% of limit</span>
                  <Switch defaultChecked />
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm">Block unsafe predictions</span>
                  <Switch defaultChecked={false} />
                </div>
              </div>
            </div>

            <Separator />

            <div className="p-3 rounded-lg bg-muted/50 border border-border/30">
              <h4 className="font-medium text-sm mb-2">Compliance Information</h4>
              <p className="text-xs text-muted-foreground">
                This tool provides estimates for research purposes. Always verify results with certified testing
                equipment for regulatory compliance.
              </p>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* System Information */}
      <Card className="border-border/50 bg-gradient-to-br from-card to-muted/20">
        <CardHeader>
          <CardTitle>System Information</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <Label className="text-muted-foreground">Version</Label>
              <p className="font-mono">v2.1.0</p>
            </div>
            <div>
              <Label className="text-muted-foreground">Last Updated</Label>
              <p className="font-mono">2024-01-15</p>
            </div>
            <div>
              <Label className="text-muted-foreground">API Status</Label>
              <p className="text-green-600 font-medium">Connected</p>
            </div>
            <div>
              <Label className="text-muted-foreground">License</Label>
              <p>Professional</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
