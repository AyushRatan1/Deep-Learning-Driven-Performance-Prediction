"use client"

import React, { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Loader2, Send, Bot, User } from "lucide-react"
import { useSAR } from "./sar-context"
import { apiClient } from "@/lib/api-client"
import type { ChatMessage } from "./sar-context"

function ChatInterface() {
  const { state, dispatch } = useSAR()
  const [message, setMessage] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const scrollAreaRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Scroll to bottom when messages change
  useEffect(() => {
    // Auto-scroll to bottom when new messages are added
    if (scrollAreaRef.current) {
      scrollAreaRef.current.scrollTop = scrollAreaRef.current.scrollHeight
    }
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [state.chatMessages])

  const handleSendMessage = async () => {
    if (!message.trim() || isLoading) return

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      message: message.trim(),
      response: "",
      timestamp: new Date().toISOString(),
      isUser: true,
    }

    dispatch({ type: "ADD_CHAT_MESSAGE", payload: userMessage })
    setMessage("")
    setIsLoading(true)

    try {
      // Provide detailed context to the API
      const response = await apiClient.chatWithAI({
        message: message.trim(),
        context: {
          currentBand: state.selectedBand,
          parameters: state.parameters,
          currentPrediction: state.currentPrediction,
          predictionHistory: state.predictionHistory.slice(0, 3), // Include last 3 predictions for context
          frequencyBands: state.frequencyBands,
          inputMode: state.inputMode,
          appState: {
            currentView: state.view,
            totalPredictions: state.predictionHistory.length,
            isSystemHealthy: state.systemStatus?.status === "healthy"
          }
        }
      })

      // Process the response - remove markdown formatting if present
      let formattedResponse = response.response;
      
      // Format the response for better readability
      formattedResponse = formattedResponse
        .replace(/\*\*(.*?)\*\*/g, "$1") // Remove bold markdown
        .replace(/\*(.*?)\*/g, "$1")     // Remove italic markdown
        .replace(/#{1,6}\s+/g, "")       // Remove heading markdown
        .replace(/`([^`]+)`/g, "$1")     // Remove inline code formatting
        .replace(/```[a-z]*\n([\s\S]*?)```/g, "$1") // Remove code blocks
      
      const aiMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        message: message.trim(),
        response: formattedResponse,
        timestamp: response.timestamp,
        isUser: false,
      }

      dispatch({ type: "ADD_CHAT_MESSAGE", payload: aiMessage })
    } catch (error) {
      console.error("Chat error:", error)
      const errorMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        message: message.trim(),
        response: "Sorry, I encountered an error processing your request. Please try again.",
        timestamp: new Date().toISOString(),
        isUser: false,
      }
      dispatch({ type: "ADD_CHAT_MESSAGE", payload: errorMessage })
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  return (
    <Card className="h-full flex flex-col">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Bot className="h-5 w-5" />
          Antenna Design Assistant
        </CardTitle>
        <p className="text-sm text-muted-foreground">
          Ask questions about antenna design, SAR calculations, or get help with your current parameters.
        </p>
      </CardHeader>
      <CardContent className="flex-1 flex flex-col gap-4">
        <ScrollArea ref={scrollAreaRef} className="flex-1 pr-4">
          <div className="space-y-4">
            {state.chatMessages.length === 0 && (
              <div className="text-center text-muted-foreground py-8">
                <Bot className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>Welcome! I'm here to help with antenna design questions.</p>
                <p className="text-sm mt-2">
                  Try asking about SAR calculations, frequency bands, or antenna parameters.
                </p>
              </div>
            )}
            {state.chatMessages.map((msg) => (
              <div
                key={msg.id}
                className={`flex gap-3 ${msg.isUser ? "flex-row-reverse" : "flex-row"}`}
              >
                <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
                  msg.isUser ? "bg-primary text-primary-foreground" : "bg-secondary"
                }`}>
                  {msg.isUser ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
                </div>
                <div className={`flex-1 space-y-2 ${msg.isUser ? "text-right" : "text-left"}`}>
                  {msg.isUser && (
                    <div className="bg-primary text-primary-foreground px-4 py-2 rounded-lg max-w-[80%] ml-auto">
                      {msg.message}
                    </div>
                  )}
                  {!msg.isUser && msg.response && (
                    <div className="bg-secondary px-4 py-2 rounded-lg max-w-[80%]">
                      <div className="whitespace-pre-wrap">{msg.response}</div>
                    </div>
                  )}
                  <div className="text-xs text-muted-foreground">
                    {new Date(msg.timestamp).toLocaleTimeString()}
                  </div>
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="flex gap-3">
                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-secondary flex items-center justify-center">
                  <Bot className="h-4 w-4" />
                </div>
                <div className="bg-secondary px-4 py-2 rounded-lg">
                  <div className="flex items-center gap-2">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    <span>Thinking...</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </ScrollArea>
        
        <div className="flex gap-2">
          <Input
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask about antenna design, SAR calculations, or get parameter suggestions..."
            disabled={isLoading}
            className="flex-1"
          />
          <Button
            onClick={handleSendMessage}
            disabled={!message.trim() || isLoading}
            size="sm"
          >
            {isLoading ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Send className="h-4 w-4" />
            )}
          </Button>
        </div>
        
        {state.selectedBand && (
          <div className="text-xs text-muted-foreground border-t pt-2">
            <strong>Current Context:</strong> {state.selectedBand.name} band
            {state.currentPrediction && ` • SAR: ${state.currentPrediction.sar_value.toFixed(3)} W/kg`}
          </div>
        )}
      </CardContent>
    </Card>
  )
}

export default ChatInterface
