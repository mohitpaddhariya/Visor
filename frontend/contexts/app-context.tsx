"use client"

import React, { createContext, useContext, useReducer, useRef, ReactNode, useEffect } from "react"
import { ParsedData } from "@/lib/constants"
import { 
  uploadPdfForOcr, 
  parseApiError, 
  getAvailableModels,
  processPdfWithStreaming
} from "@/lib/api"

// Model info interface
interface ModelInfo {
  name: string
  description: string
  supports_bounding_boxes: boolean
  features: string[]
}

// State interface
interface AppState {
  uploadedFile: File | null
  isProcessing: boolean
  parsedData: ParsedData | null
  hoveredBbox: string | null
  error: string | null
  processingFile: string
  selectedModel: string
  availableModels: ModelInfo[]
  isLoadingModels: boolean
  dpi: number
  enableAnnotations: boolean
  jobId: string | null  // Track current job ID
  processingStatus: string  // Track job status (pending, running, completed, etc.)
  processingProgress: string  // User-friendly progress message
}

// Action types
type AppAction =
  | { type: "SET_UPLOADED_FILE"; payload: File | null }
  | { type: "SET_PROCESSING"; payload: boolean }
  | { type: "SET_PARSED_DATA"; payload: ParsedData | null }
  | { type: "SET_HOVERED_BBOX"; payload: string | null }
  | { type: "SET_ERROR"; payload: string | null }
  | { type: "SET_PROCESSING_FILE"; payload: string }
  | { type: "SET_SELECTED_MODEL"; payload: string }
  | { type: "SET_AVAILABLE_MODELS"; payload: ModelInfo[] }
  | { type: "SET_LOADING_MODELS"; payload: boolean }
  | { type: "SET_DPI"; payload: number }
  | { type: "SET_ENABLE_ANNOTATIONS"; payload: boolean }
  | { type: "SET_JOB_ID"; payload: string | null }
  | { type: "SET_PROCESSING_STATUS"; payload: string }
  | { type: "SET_PROCESSING_PROGRESS"; payload: string }
  | { type: "START_OCR"; payload: string }
  | { type: "OCR_SUCCESS"; payload: ParsedData }
  | { type: "OCR_ERROR"; payload: string }
  | { type: "OCR_CANCELLED" }
  | { type: "RESET" }

// Initial state
const initialState: AppState = {
  uploadedFile: null,
  isProcessing: false,
  parsedData: null,
  hoveredBbox: null,
  error: null,
  processingFile: "",
  selectedModel: "dotsocr", // Default model
  availableModels: [],
  isLoadingModels: false,
  dpi: 200, // Default DPI
  enableAnnotations: false,
  jobId: null,
  processingStatus: "",
  processingProgress: "",
}

// Reducer function
function appReducer(state: AppState, action: AppAction): AppState {
  switch (action.type) {
    case "SET_UPLOADED_FILE":
      return { ...state, uploadedFile: action.payload }
    case "SET_PROCESSING":
      return { ...state, isProcessing: action.payload }
    case "SET_PARSED_DATA":
      return { ...state, parsedData: action.payload }
    case "SET_HOVERED_BBOX":
      return { ...state, hoveredBbox: action.payload }
    case "SET_ERROR":
      return { ...state, error: action.payload }
    case "SET_PROCESSING_FILE":
      return { ...state, processingFile: action.payload }
    case "SET_SELECTED_MODEL":
      return { ...state, selectedModel: action.payload }
    case "SET_AVAILABLE_MODELS":
      return { ...state, availableModels: action.payload }
    case "SET_LOADING_MODELS":
      return { ...state, isLoadingModels: action.payload }
    case "SET_DPI":
      return { ...state, dpi: action.payload }
    case "SET_ENABLE_ANNOTATIONS":
      return { ...state, enableAnnotations: action.payload }
    case "SET_JOB_ID":
      return { ...state, jobId: action.payload }
    case "SET_PROCESSING_STATUS":
      return { ...state, processingStatus: action.payload }
    case "SET_PROCESSING_PROGRESS":
      return { ...state, processingProgress: action.payload }
    case "START_OCR":
      return {
        ...state,
        isProcessing: true,
        error: null,
        processingFile: action.payload,
        processingStatus: "starting",
        processingProgress: "Submitting job...",
      }
    case "OCR_SUCCESS":
      return {
        ...state,
        isProcessing: false,
        parsedData: action.payload,
        error: null,
        processingStatus: "completed",
        processingProgress: "Processing complete!",
      }
    case "OCR_ERROR":
      return {
        ...state,
        isProcessing: false,
        error: action.payload,
        processingStatus: "error",
        processingProgress: "",
      }
    case "OCR_CANCELLED":
      return {
        ...state,
        isProcessing: false,
        error: "Processing cancelled by user",
        processingStatus: "cancelled",
        processingProgress: "",
      }
    case "RESET":
      return {
        ...initialState,
        // Preserve model settings after reset
        selectedModel: state.selectedModel,
        availableModels: state.availableModels,
        isLoadingModels: state.isLoadingModels,
        dpi: state.dpi,
        enableAnnotations: state.enableAnnotations,
      }
    default:
      return state
  }
}

// Context type
interface AppContextType extends AppState {
  setUploadedFile: (file: File | null) => void
  setIsProcessing: (processing: boolean) => void
  setParsedData: (data: ParsedData | null) => void
  setHoveredBbox: (id: string | null) => void
  setError: (error: string | null) => void
  setProcessingFile: (filename: string) => void
  setSelectedModel: (model: string) => void
  setDpi: (dpi: number) => void
  setEnableAnnotations: (enable: boolean) => void
  handleRunOcr: () => Promise<void>
  handleCancelOcr: () => void
  resetApp: () => void
  fetchAvailableModels: () => Promise<void>
}

const AppContext = createContext<AppContextType | undefined>(undefined)

export function AppProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(appReducer, initialState)
  const abortControllerRef = useRef<AbortController | null>(null)

  // Fetch available models on mount
  useEffect(() => {
    fetchAvailableModels()
  }, [])

  const fetchAvailableModels = async () => {
    dispatch({ type: "SET_LOADING_MODELS", payload: true })
    try {
      const result = await getAvailableModels()
      if (result.success && result.data) {
        // Parse the models data from the API
        const apiData = result.data as any
        
        if (apiData.models && typeof apiData.models === 'object') {
          // Transform the models object into an array with the supports_bounding_boxes flag
          const modelsArray = Object.entries(apiData.models).map(([key, value]: [string, any]) => ({
            name: key,
            description: value.description || '',
            supports_bounding_boxes: value.supports_bounding_boxes || false,
            features: value.features || []
          }))
          dispatch({ type: "SET_AVAILABLE_MODELS", payload: modelsArray })
        } else if (Array.isArray(apiData)) {
          dispatch({ type: "SET_AVAILABLE_MODELS", payload: apiData })
        } else {
          // Fallback to default models if API doesn't return expected format
          dispatch({
            type: "SET_AVAILABLE_MODELS",
            payload: [
              {
                name: "dotsocr",
                description: "Structured layout JSON with bbox, categories, and formatted text",
                supports_bounding_boxes: true,
                features: ["Bounding boxes", "Element categories", "Layout analysis"]
              },
              {
                name: "lightonocr",
                description: "Clean markdown text extraction",
                supports_bounding_boxes: false,
                features: ["Markdown output", "Fast processing", "Clean text"]
              }
            ]
          })
        }
      }
    } catch (error) {
      console.error("Failed to fetch models:", error)
      // Set default models on error
      dispatch({
        type: "SET_AVAILABLE_MODELS",
        payload: [
          {
            name: "dotsocr",
            description: "Structured layout JSON with bbox, categories, and formatted text",
            supports_bounding_boxes: true,
            features: ["Bounding boxes", "Element categories", "Layout analysis"]
          },
          {
            name: "lightonocr",
            description: "Clean markdown text extraction",
            supports_bounding_boxes: false,
            features: ["Markdown output", "Fast processing", "Clean text"]
          }
        ]
      })
    } finally {
      dispatch({ type: "SET_LOADING_MODELS", payload: false })
    }
  }

  const handleRunOcr = async () => {
    if (!state.uploadedFile) return

    dispatch({ type: "START_OCR", payload: state.uploadedFile.name })

    const controller = new AbortController()
    abortControllerRef.current = controller

    try {
      // Use the new job queue API with SSE streaming
      const result = await processPdfWithStreaming(
        state.uploadedFile,
        {
          dpi: state.dpi,
          model: state.selectedModel,
        },
        {
          onStatus: (status, elapsed) => {
            dispatch({ type: "SET_PROCESSING_STATUS", payload: status })
            
            let progressMessage = ""
            switch (status) {
              case "pending":
                progressMessage = "Job queued, waiting to start..."
                break
              case "running":
                progressMessage = `Processing... ${elapsed ? `(${elapsed}s)` : ""}`
                break
              case "completed":
                progressMessage = "Processing complete!"
                break
              default:
                progressMessage = `Status: ${status}`
            }
            
            dispatch({ type: "SET_PROCESSING_PROGRESS", payload: progressMessage })
          },
          onProgress: (message) => {
            dispatch({ type: "SET_PROCESSING_PROGRESS", payload: message })
          },
        }
      )

      if (result.success && result.data) {
        dispatch({ type: "OCR_SUCCESS", payload: result.data })
      } else {
        dispatch({ type: "OCR_ERROR", payload: result.error || "Failed to process PDF" })
        console.error("OCR processing failed:", result.error)
      }
    } catch (err) {
      const errorMessage = parseApiError(err)
      dispatch({ type: "OCR_ERROR", payload: errorMessage })
      console.error("OCR processing error:", err)
    } finally {
      abortControllerRef.current = null
    }
  }

  const handleCancelOcr = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      dispatch({ type: "OCR_CANCELLED" })
      abortControllerRef.current = null
    }
  }

  const resetApp = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      abortControllerRef.current = null
    }
    dispatch({ type: "RESET" })
  }

  const value: AppContextType = {
    ...state,
    setUploadedFile: (file) => dispatch({ type: "SET_UPLOADED_FILE", payload: file }),
    setIsProcessing: (processing) => dispatch({ type: "SET_PROCESSING", payload: processing }),
    setParsedData: (data) => dispatch({ type: "SET_PARSED_DATA", payload: data }),
    setHoveredBbox: (id) => dispatch({ type: "SET_HOVERED_BBOX", payload: id }),
    setError: (error) => dispatch({ type: "SET_ERROR", payload: error }),
    setProcessingFile: (filename) => dispatch({ type: "SET_PROCESSING_FILE", payload: filename }),
    setSelectedModel: (model) => dispatch({ type: "SET_SELECTED_MODEL", payload: model }),
    setDpi: (dpi) => dispatch({ type: "SET_DPI", payload: dpi }),
    setEnableAnnotations: (enable) => dispatch({ type: "SET_ENABLE_ANNOTATIONS", payload: enable }),
    handleRunOcr,
    handleCancelOcr,
    resetApp,
    fetchAvailableModels,
  }

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>
}

export function useApp() {
  const context = useContext(AppContext)
  if (context === undefined) {
    throw new Error("useApp must be used within an AppProvider")
  }
  return context
}
