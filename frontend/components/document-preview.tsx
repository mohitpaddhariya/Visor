"use client"

import type React from "react"
import { useEffect, useRef, useState, useCallback } from "react"
import { Upload } from "lucide-react"
import { BoundingBox, CATEGORY_COLOR_MAP, hexToRgba } from "@/lib/constants"
import { useApp } from "@/contexts/app-context"

export default function DocumentPreview() {
  const {
    isProcessing,
    uploadedFile,
    parsedData,
    hoveredBbox,
    setHoveredBbox,
    setUploadedFile,
    availableModels,
  } = useApp()

  const canvasRefs = useRef<Map<number, HTMLCanvasElement>>(new Map())
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [numPages, setNumPages] = useState<number>(0)
  const containerRef = useRef<HTMLDivElement>(null)
  const [pdfDocument, setPdfDocument] = useState<any>(null)
  const [pageImages, setPageImages] = useState<Map<number, ImageData>>(new Map())
  const [pageSizes, setPageSizes] = useState<Map<number, { widthInches: number; heightInches: number }>>(new Map())
  const [renderScale, setRenderScale] = useState<number>(2.0)
  const [canvasWidth, setCanvasWidth] = useState<number>(800)
  const [pdfjsLib, setPdfjsLib] = useState<any>(null)

  // Dynamically import pdfjs-dist only on client side
  useEffect(() => {
    const loadPdfJs = async () => {
      const pdfjs = await import("pdfjs-dist")
      pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`
      setPdfjsLib(pdfjs)
    }
    loadPdfJs()
  }, [])

  const handleFileUpload = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setUploadedFile(file)
    }
  }, [setUploadedFile])

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    const file = e.dataTransfer.files?.[0]
    if (file) {
      setUploadedFile(file)
    }
  }, [setUploadedFile])

  // Resize observer to adjust canvas width
  useEffect(() => {
    if (!containerRef.current) return
    
    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const width = entry.contentRect.width
        setCanvasWidth(Math.min(width - 40, 1000))
      }
    })
    
    resizeObserver.observe(containerRef.current)
    return () => resizeObserver.disconnect()
  }, [])

  // Render a single page at given scale
  const renderPage = useCallback(async (pdf: any, pageNum: number, scale: number): Promise<ImageData> => {
    const page = await pdf.getPage(pageNum)
    
    const pageWidthPoints = page.view[2] - page.view[0]
    const pageHeightPoints = page.view[3] - page.view[1]
    const widthInches = pageWidthPoints / 72
    const heightInches = pageHeightPoints / 72
    
    setPageSizes(prev => {
      if (prev.get(pageNum)) return prev
      return new Map(prev).set(pageNum, { widthInches, heightInches })
    })
    
    const viewport = page.getViewport({ scale })
    const tempCanvas = document.createElement('canvas')
    const context = tempCanvas.getContext('2d', { alpha: false })
    if (!context) throw new Error('Failed to get canvas context')
    
    tempCanvas.width = viewport.width
    tempCanvas.height = viewport.height
    
    await page.render({
      canvasContext: context,
      viewport: viewport
    }).promise
    
    return context.getImageData(0, 0, tempCanvas.width, tempCanvas.height)
  }, [])

  // Load PDF and convert pages to images
  useEffect(() => {
    if (!uploadedFile || !pdfjsLib) {
      setPdfDocument(null)
      setPageImages(new Map())
      setPageSizes(new Map())
      setNumPages(0)
      return
    }

    const loadPdf = async () => {
      try {
        const fileUrl = URL.createObjectURL(uploadedFile)
        const loadingTask = pdfjsLib.getDocument(fileUrl)
        const pdf = await loadingTask.promise
        
        setPdfDocument(pdf)
        setNumPages(pdf.numPages)

        const images = new Map<number, ImageData>()
        for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {
          const imageData = await renderPage(pdf, pageNum, renderScale)
          images.set(pageNum, imageData)
        }
        
        setPageImages(images)
        URL.revokeObjectURL(fileUrl)
      } catch (error) {
        console.error("Error loading PDF:", error)
      }
    }

    loadPdf()
  }, [uploadedFile, pdfjsLib, renderScale, renderPage])

  // Re-render pages at backend DPI when parsedData arrives
  useEffect(() => {
    if (!pdfDocument || !parsedData || pageImages.size === 0) return
    
    const backendDpi = parsedData.dpi || 200
    const newScale = backendDpi / 72
    if (Math.abs(newScale - renderScale) < 0.01) return
    
    const reRenderPages = async () => {
      const images = new Map<number, ImageData>()
      for (let pageNum = 1; pageNum <= numPages; pageNum++) {
        const imageData = await renderPage(pdfDocument, pageNum, newScale)
        images.set(pageNum, imageData)
      }
      setPageImages(images)
      setRenderScale(newScale)
    }
    
    reRenderPages()
  }, [parsedData, pdfDocument, pageImages.size, renderScale, numPages, renderPage])

  const drawPageWithBoundingBoxes = useCallback((pageNum: number) => {
    const canvas = canvasRefs.current.get(pageNum)
    if (!canvas) return

    const ctx = canvas.getContext("2d", { alpha: false })
    if (!ctx) return

    const imageData = pageImages.get(pageNum)
    const pageSize = pageSizes.get(pageNum)
    if (!imageData || !pageSize) return

    const aspectRatio = imageData.height / imageData.width
    const displayWidth = canvasWidth
    const displayHeight = displayWidth * aspectRatio

    canvas.style.width = `${displayWidth}px`
    canvas.style.height = `${displayHeight}px`
    canvas.width = imageData.width
    canvas.height = imageData.height

    ctx.putImageData(imageData, 0, 0)

    if (!parsedData?.results_by_page) return

    // Check if the current model supports bounding boxes
    const currentModel = parsedData.model
    const modelInfo = availableModels.find(m => m.name === currentModel)
    const supportsBoundingBoxes = modelInfo?.supports_bounding_boxes ?? true // Default to true for backward compatibility

    // Only draw bounding boxes if the model supports them
    if (!supportsBoundingBoxes) return

    const pageKey = `page_${pageNum - 1}`
    const pageResult = parsedData.results_by_page[pageKey]

    if (!pageResult || pageResult.status !== "success" || !pageResult.data) return

    const backendDpi = parsedData.dpi || 200
    const backendImageWidth = pageSize.widthInches * backendDpi
    const backendImageHeight = pageSize.heightInches * backendDpi
    
    const scaleX = canvas.width / backendImageWidth
    const scaleY = canvas.height / backendImageHeight

    pageResult.data.forEach((item: BoundingBox, index: number) => {
      const [x1, y1, x2, y2] = item.bbox
      const bboxId = `${pageNum - 1}-${item.category}-${index}`
      const isHovered = hoveredBbox === bboxId
      const categoryColor = CATEGORY_COLOR_MAP[item.category] || "#91c9ef"

      const scaledX1 = x1 * scaleX
      const scaledY1 = y1 * scaleY
      const scaledX2 = x2 * scaleX
      const scaledY2 = y2 * scaleY

      ctx.fillStyle = isHovered ? hexToRgba(categoryColor, 0.3) : hexToRgba(categoryColor, 0.15)
      ctx.fillRect(scaledX1, scaledY1, scaledX2 - scaledX1, scaledY2 - scaledY1)

      ctx.strokeStyle = categoryColor
      ctx.lineWidth = isHovered ? 3 : 2
      ctx.strokeRect(scaledX1, scaledY1, scaledX2 - scaledX1, scaledY2 - scaledY1)

      if (isHovered) {
        const label = item.category
        const padding = 6
        const fontSize = 16
        ctx.font = `bold ${fontSize}px sans-serif`
        const textMetrics = ctx.measureText(label)
        const labelWidth = textMetrics.width + padding * 2
        const labelHeight = fontSize + padding * 2

        ctx.fillStyle = categoryColor
        ctx.fillRect(scaledX1, scaledY1 - labelHeight, labelWidth, labelHeight)

        ctx.fillStyle = "#ffffff"
        ctx.textBaseline = "top"
        ctx.fillText(label, scaledX1 + padding, scaledY1 - labelHeight + padding)
      }
    })
  }, [pageImages, pageSizes, canvasWidth, parsedData, hoveredBbox, availableModels])

  // Redraw all pages when data changes
  useEffect(() => {
    if (pageImages.size > 0) {
      const timer = setTimeout(() => {
        for (let i = 1; i <= numPages; i++) {
          drawPageWithBoundingBoxes(i)
        }
      }, 100)
      return () => clearTimeout(timer)
    }
  }, [pageImages, parsedData, canvasWidth, numPages, drawPageWithBoundingBoxes])

  // Immediate redraw on hover
  useEffect(() => {
    if (pageImages.size > 0) {
      for (let i = 1; i <= numPages; i++) {
        drawPageWithBoundingBoxes(i)
      }
    }
  }, [hoveredBbox, pageImages.size, numPages, drawPageWithBoundingBoxes])

  const handleCanvasMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>, pageNum: number) => {
    if (!parsedData?.results_by_page) return
    
    // Check if the current model supports bounding boxes
    const currentModel = parsedData.model
    const modelInfo = availableModels.find(m => m.name === currentModel)
    const supportsBoundingBoxes = modelInfo?.supports_bounding_boxes ?? true
    
    // Only handle hover if the model supports bounding boxes
    if (!supportsBoundingBoxes) return
    
    const canvas = e.currentTarget
    const rect = canvas.getBoundingClientRect()
    const x = (e.clientX - rect.left) * (canvas.width / rect.width)
    const y = (e.clientY - rect.top) * (canvas.height / rect.height)

    const pageKey = `page_${pageNum - 1}`
    const pageResult = parsedData.results_by_page[pageKey]
    
    if (pageResult?.status === "success" && pageResult.data) {
      const pageSize = pageSizes.get(pageNum)
      if (!pageSize) return
      
      const backendDpi = parsedData.dpi || 200
      const backendImageWidth = pageSize.widthInches * backendDpi
      const backendImageHeight = pageSize.heightInches * backendDpi
      
      const scaleX = canvas.width / backendImageWidth
      const scaleY = canvas.height / backendImageHeight
      
      let found = false
      pageResult.data.forEach((item: BoundingBox, index: number) => {
        const [x1, y1, x2, y2] = item.bbox
        const scaledX1 = x1 * scaleX
        const scaledY1 = y1 * scaleY
        const scaledX2 = x2 * scaleX
        const scaledY2 = y2 * scaleY
        
        if (x >= scaledX1 && x <= scaledX2 && y >= scaledY1 && y <= scaledY2) {
          setHoveredBbox(`${pageNum - 1}-${item.category}-${index}`)
          found = true
        }
      })
      if (!found) setHoveredBbox(null)
    }
  }, [parsedData, pageSizes, setHoveredBbox, availableModels])

  return (
    <div
      className="w-full h-full bg-card overflow-auto p-4 sm:p-8"
      onDragOver={handleDragOver}
      onDrop={handleDrop}
    >
      <div className="min-h-full flex items-center justify-center">
        {!uploadedFile ? (
          <div
            onClick={() => fileInputRef.current?.click()}
            className="text-center cursor-pointer hover:opacity-80 transition-opacity p-4"
          >
            <div className="w-16 h-16 sm:w-24 sm:h-24 bg-muted rounded-lg flex items-center justify-center mx-auto mb-4">
              <Upload className="w-8 h-8 sm:w-12 sm:h-12 text-muted-foreground" />
            </div>
            <p className="text-foreground font-medium mb-1 text-sm sm:text-base">Upload a document</p>
            <p className="text-xs sm:text-sm text-muted-foreground">Drag and drop or click to select a PDF</p>
            <input
              ref={fileInputRef}
              type="file"
              onChange={handleFileUpload}
              className="hidden"
              accept=".pdf,.png,.jpg,.jpeg"
            />
          </div>
        ) : isProcessing ? (
          <div className="text-center p-4">
            <div className="w-12 h-12 sm:w-16 sm:h-16 border-4 border-accent-magenta border-t-transparent rounded-full animate-spin mx-auto mb-4" />
            <p className="text-foreground font-medium mb-2 text-sm sm:text-base">Processing document...</p>
            <p className="text-xs sm:text-sm text-muted-foreground truncate max-w-xs">{uploadedFile?.name || "Loading..."}</p>
          </div>
        ) : pageImages.size > 0 ? (
          <div className="w-full max-w-5xl" ref={containerRef}>
            {Array.from(new Array(numPages), (el, index) => {
              const pageNum = index + 1
              return (
                <div key={`page_${pageNum}`} className="mb-4 sm:mb-5">
                  <canvas
                    ref={(el) => {
                      if (el) {
                        canvasRefs.current.set(pageNum, el)
                        if (pageImages.get(pageNum)) {
                          drawPageWithBoundingBoxes(pageNum)
                        }
                      } else {
                        canvasRefs.current.delete(pageNum)
                      }
                    }}
                    className="w-full shadow-lg cursor-pointer rounded-md"
                    onMouseMove={(e) => handleCanvasMouseMove(e, pageNum)}
                    onMouseLeave={() => setHoveredBbox(null)}
                  />
                </div>
              )
            })}
          </div>
        ) : (
          <div className="text-center p-4">
            <div className="w-12 h-12 sm:w-16 sm:h-16 border-4 border-accent-magenta border-t-transparent rounded-full animate-spin mx-auto mb-4" />
            <p className="text-foreground font-medium mb-2 text-sm sm:text-base">Loading document...</p>
          </div>
        )}
      </div>
    </div>
  )
}
