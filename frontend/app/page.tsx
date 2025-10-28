"use client"

import { Moon, Sun, RotateCcw } from "lucide-react"
import { useTheme } from "next-themes"
import { Button } from "@/components/ui/button"
import DocumentPreview from "@/components/document-preview"
import ResultsPanel from "@/components/results-panel"
import { useApp } from "@/contexts/app-context"

export default function Home() {
  const { theme, setTheme } = useTheme()
  const { resetApp, uploadedFile } = useApp()

  const toggleTheme = () => {
    setTheme(theme === "dark" ? "light" : "dark")
  }

  const handleReset = () => {
    if (confirm("Are you sure you want to start over? This will clear all current work.")) {
      resetApp()
    }
  }

  return (
    <div className="flex flex-col h-screen bg-background text-foreground">
      {/* Header */}
      <header className="border-b border-border px-4 sm:px-6 py-3 sm:py-4 flex items-center justify-between flex-shrink-0">
        <h1 className="text-lg sm:text-xl font-semibold">Visor</h1>

        <div className="flex items-center gap-1 sm:gap-2">
          {uploadedFile && (
            <Button 
              onClick={handleReset}
              variant="ghost"
              size="icon"
              aria-label="Reset and start over"
              title="Reset"
              className="h-8 w-8 sm:h-10 sm:w-10"
            >
              <RotateCcw className="w-4 h-4 sm:w-5 sm:h-5" />
            </Button>
          )}
          <Button 
            onClick={toggleTheme}
            variant="ghost"
            size="icon"
            aria-label="Toggle theme"
            className="h-8 w-8 sm:h-10 sm:w-10"
          >
            {theme === "dark" ? (
              <Sun className="w-4 h-4 sm:w-5 sm:h-5" />
            ) : (
              <Moon className="w-4 h-4 sm:w-5 sm:h-5" />
            )}
          </Button>
        </div>
      </header>

      {/* Main Content - Responsive Layout */}
      <div className="flex-1 flex flex-col lg:flex-row overflow-hidden">
        {/* Document Preview */}
        <div className="w-full lg:w-1/2 h-1/2 lg:h-full border-b lg:border-b-0 lg:border-r border-border">
          <DocumentPreview />
        </div>

        {/* Right Panel - Results */}
        <div className="w-full lg:w-1/2 h-1/2 lg:h-full">
          <ResultsPanel />
        </div>
      </div>
    </div>
  )
}
