"use client"

import { SidebarProvider } from "@/components/ui/sidebar"
import { AppSidebar } from "@/components/app-sidebar"
import DashboardContent from "@/components/dashboard-content"
import { SARProvider } from "@/components/sar-context"

export default function Dashboard() {
  return (
    <SARProvider>
      <SidebarProvider defaultOpen={true}>
        <div className="min-h-screen flex w-full bg-gradient-to-br from-background via-background to-muted/20">
          <AppSidebar />
          <DashboardContent />
        </div>
      </SidebarProvider>
    </SARProvider>
  )
}
