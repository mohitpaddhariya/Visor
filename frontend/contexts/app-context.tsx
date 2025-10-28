"use client"

import React, { createContext, useContext, useReducer, useRef, ReactNode } from "react"
import { ParsedData } from "@/lib/constants"
import { uploadPdfForOcr, parseApiError } from "@/lib/api"

// State interface
interface AppState {
  uploadedFile: File | null
  isProcessing: boolean
  parsedData: ParsedData | null
  hoveredBbox: string | null
  error: string | null
  processingFile: string
}

// Action types
type AppAction =
  | { type: "SET_UPLOADED_FILE"; payload: File | null }
  | { type: "SET_PROCESSING"; payload: boolean }
  | { type: "SET_PARSED_DATA"; payload: ParsedData | null }
  | { type: "SET_HOVERED_BBOX"; payload: string | null }
  | { type: "SET_ERROR"; payload: string | null }
  | { type: "SET_PROCESSING_FILE"; payload: string }
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
    case "START_OCR":
      return {
        ...state,
        isProcessing: true,
        error: null,
        processingFile: action.payload,
      }
    case "OCR_SUCCESS":
      return {
        ...state,
        isProcessing: false,
        parsedData: action.payload,
        error: null,
      }
    case "OCR_ERROR":
      return {
        ...state,
        isProcessing: false,
        error: action.payload,
      }
    case "OCR_CANCELLED":
      return {
        ...state,
        isProcessing: false,
        error: "Processing cancelled by user",
      }
    case "RESET":
      return initialState
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
  handleRunOcr: () => Promise<void>
  handleCancelOcr: () => void
  resetApp: () => void
}

const AppContext = createContext<AppContextType | undefined>(undefined)

export function AppProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(appReducer, initialState)
  const abortControllerRef = useRef<AbortController | null>(null)

  const handleRunOcr = async () => {
    if (!state.uploadedFile) return

    dispatch({ type: "START_OCR", payload: state.uploadedFile.name })

    const controller = new AbortController()
    abortControllerRef.current = controller

    try {
      const result = await uploadPdfForOcr(state.uploadedFile, {
        dpi: 200, // Default DPI, can be made configurable
      })

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
    handleRunOcr,
    handleCancelOcr,
    resetApp,
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
