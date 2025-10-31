"use client"

import React from "react"
import { useApp } from "@/contexts/app-context"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Separator } from "@/components/ui/separator"

export function SettingsPanel() {
  const {
    selectedModel,
    setSelectedModel,
    availableModels,
    isLoadingModels,
  } = useApp()

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>OCR Settings</CardTitle>
        <CardDescription>
          Configure OCR processing options
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Model Selection */}
        <div className="space-y-2">
          <label htmlFor="model-select" className="text-sm font-medium">
            OCR Model
          </label>
          <Select
            value={selectedModel}
            onValueChange={setSelectedModel}
            disabled={isLoadingModels}
          >
            <SelectTrigger id="model-select">
              <SelectValue placeholder="Select a model" />
            </SelectTrigger>
            <SelectContent>
              {availableModels.map((model) => (
                <SelectItem key={model.name} value={model.name}>
                  <div className="flex flex-col">
                    <span className="font-medium">{model.name}</span>
                    <span className="text-xs text-muted-foreground">
                      {model.description}
                    </span>
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          {selectedModel && (
            <div className="text-xs text-muted-foreground mt-2">
              {availableModels.find((m) => m.name === selectedModel)?.description}
            </div>
          )}
        </div>

        {/* Model Features */}
        {selectedModel && (
          <>
            <Separator />
            <div className="space-y-2">
              <p className="text-sm font-medium">Model Features</p>
              <ul className="list-disc list-inside text-xs text-muted-foreground space-y-1">
                {availableModels
                  .find((m) => m.name === selectedModel)
                  ?.features.map((feature, index) => (
                    <li key={index}>{feature}</li>
                  ))}
              </ul>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  )
}
