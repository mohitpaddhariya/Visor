"use client"

import { useEffect, useRef, useCallback, useMemo } from "react"
import { Loader2, Copy, Download, FileJson } from "lucide-react"
import { Button } from "@/components/ui/button"
import { BoundingBox, PageResult } from "@/lib/constants"
import { useApp } from "@/contexts/app-context"
import { SettingsPanel } from "@/components/settings-panel"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import rehypeRaw from "rehype-raw"
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter"
import { tomorrow } from "react-syntax-highlighter/dist/esm/styles/prism"

export default function ResultsPanel() {
  const {
    isProcessing,
    parsedData,
    hoveredBbox,
    uploadedFile,
    error,
    processingFile,
    handleRunOcr,
    handleCancelOcr,
    availableModels,
  } = useApp()

  const contentRef = useRef<HTMLDivElement>(null)

  // Check if current model supports bounding boxes
  const supportsBoundingBoxes = useMemo(() => {
    if (!parsedData?.model) return true // Default to true for backward compatibility
    const modelInfo = availableModels.find(m => m.name === parsedData.model)
    return modelInfo?.supports_bounding_boxes ?? true
  }, [parsedData?.model, availableModels])

  // Type for processed items
  type ProcessedItem = BoundingBox & { id: string; pageNum: number }

  // Group and sort data from all pages - memoized for performance
  const groupedData = useMemo(() => {
    if (!parsedData?.results_by_page) return []
    
    // If model doesn't support bounding boxes, data is likely a string
    if (!supportsBoundingBoxes) {
      return Object.entries(parsedData.results_by_page)
        .sort(([keyA], [keyB]) => {
          const numA = parseInt(keyA.split("_")[1])
          const numB = parseInt(keyB.split("_")[1])
          return numA - numB
        })
        .flatMap(([pageKey, page]: [string, PageResult]) => {
          if (page.status === "success" && page.data) {
            const pageNum = parseInt(pageKey.split("_")[1])
            // For string data (markdown), create a single item
            if (typeof page.data === 'string') {
              return [{
                text: page.data,
                id: `${pageNum}-text-0`,
                pageNum,
                category: 'Text',
                bbox: [0, 0, 0, 0] as [number, number, number, number]
              }] as ProcessedItem[]
            }
          }
          return []
        })
    }
    
    // For models with bounding boxes (array data)
    return Object.entries(parsedData.results_by_page)
      .sort(([keyA], [keyB]) => {
        const numA = parseInt(keyA.split("_")[1])
        const numB = parseInt(keyB.split("_")[1])
        return numA - numB
      })
      .flatMap(([pageKey, page]: [string, PageResult]) => {
        if (page.status === "success" && page.data && Array.isArray(page.data)) {
          const pageNum = parseInt(pageKey.split("_")[1])
          return page.data
            // Don't filter out items - show everything including empty text
            .map((item: BoundingBox, index: number) => ({
              ...item,
              id: `${pageNum}-${item.category}-${index}`,
              pageNum,
            })) as ProcessedItem[]
        }
        return []
      })
  }, [parsedData, supportsBoundingBoxes])

  // Group by page for better structure - memoized
  const groupedByPage = useMemo(() => {
    if (!parsedData?.results_by_page) return []
    
    console.log('Processing parsedData:', parsedData)
    console.log('Supports bounding boxes:', supportsBoundingBoxes)
    
    // If model doesn't support bounding boxes, data is likely a string
    if (!supportsBoundingBoxes) {
      return Object.entries(parsedData.results_by_page)
        .sort(([keyA], [keyB]) => {
          const numA = parseInt(keyA.split("_")[1])
          const numB = parseInt(keyB.split("_")[1])
          return numA - numB
        })
        .map(([pageKey, page]: [string, PageResult]) => {
          if (page.status === "success" && page.data) {
            const pageNum = parseInt(pageKey.split("_")[1])
            // For string data (markdown), create a single item
            if (typeof page.data === 'string') {
              return {
                pageNum,
                items: [{
                  text: page.data,
                  id: `${pageNum}-text-0`,
                  pageNum,
                  category: 'Text',
                  bbox: [0, 0, 0, 0] as [number, number, number, number]
                }] as ProcessedItem[]
              }
            }
          }
          return null
        })
        .filter((item): item is { pageNum: number; items: ProcessedItem[] } => item !== null)
    }
    
    // For models with bounding boxes (array data)
    return Object.entries(parsedData.results_by_page)
      .sort(([keyA], [keyB]) => {
        const numA = parseInt(keyA.split("_")[1])
        const numB = parseInt(keyB.split("_")[1])
        return numA - numB
      })
      .map(([pageKey, page]: [string, PageResult]) => {
        if (page.status === "success" && page.data && Array.isArray(page.data)) {
          const pageNum = parseInt(pageKey.split("_")[1])
          console.log(`Page ${pageNum} raw data:`, page.data)
          console.log(`Page ${pageNum} first item structure:`, page.data[0])
          const pageItems = page.data
            // Don't filter out items - show everything including empty text
            .map((item: BoundingBox, index: number) => ({
              ...item,
              id: `${pageNum}-${item.category}-${index}`,
              pageNum,
            })) as ProcessedItem[]
          console.log(`Page ${pageNum} processed items:`, pageItems)
          console.log(`Page ${pageNum} first processed item:`, pageItems[0])
          return { pageNum, items: pageItems }
        }
        return null
      })
      .filter((item): item is { pageNum: number; items: ProcessedItem[] } => item !== null)
  }, [parsedData, supportsBoundingBoxes])

  const handleCopy = useCallback(() => {
    if (!parsedData) return
    const text = groupedData.map(item => item.text || '').filter(t => t).join('\n\n')
    navigator.clipboard.writeText(text)
  }, [parsedData, groupedData])

  const handleDownload = useCallback(() => {
    if (!parsedData) return
    const text = groupedData.map(item => item.text || '').filter(t => t).join('\n\n')
    const blob = new Blob([text], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${parsedData.filename.replace('.pdf', '')}_parsed.txt`
    a.click()
    URL.revokeObjectURL(url)
  }, [parsedData, groupedData])

  const handleDownloadJson = useCallback(() => {
    if (!parsedData) return
    const blob = new Blob([JSON.stringify(parsedData, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${parsedData.filename.replace('.pdf', '')}_parsed.json`
    a.click()
    URL.revokeObjectURL(url)
  }, [parsedData])

  // Scroll to hovered item
  useEffect(() => {
    if (hoveredBbox && contentRef.current) {
      const element = document.getElementById(`item-${hoveredBbox}`)
      if (element) {
        element.scrollIntoView({ behavior: "smooth", block: "center" })
      }
    }
  }, [hoveredBbox])

  // Custom components for ReactMarkdown to style elements
  const markdownComponents = useMemo(() => ({
    h1: ({ children }: { children?: React.ReactNode }) => (
      <h1 className="text-xl sm:text-2xl font-bold mt-6 sm:mt-8 mb-3 sm:mb-4 text-foreground border-b border-border pb-2">
        {children}
      </h1>
    ),
    h2: ({ children }: { children?: React.ReactNode }) => (
      <h2 className="text-lg sm:text-xl font-semibold mt-5 sm:mt-6 mb-2 sm:mb-3 text-foreground">
        {children}
      </h2>
    ),
    h3: ({ children }: { children?: React.ReactNode }) => (
      <h3 className="text-base sm:text-lg font-medium mt-4 sm:mt-5 mb-2 text-foreground">
        {children}
      </h3>
    ),
    p: ({ children }: { children?: React.ReactNode }) => (
      <p className="mb-3 text-xs sm:text-sm leading-relaxed text-foreground/90 whitespace-pre-wrap">
        {children}
      </p>
    ),
    ul: ({ children }: { children?: React.ReactNode }) => (
      <ul className="mb-3 ml-4 sm:ml-6 list-disc space-y-1">
        {children}
      </ul>
    ),
    ol: ({ children }: { children?: React.ReactNode }) => (
      <ol className="mb-3 ml-4 sm:ml-6 list-decimal space-y-1">
        {children}
      </ol>
    ),
    li: ({ children }: { children?: React.ReactNode }) => (
      <li className="text-xs sm:text-sm leading-relaxed text-foreground/90">
        {children}
      </li>
    ),
    code: ({ children, className }: { children?: React.ReactNode; className?: string }) => {
      const inline = !className?.includes("language-")
      if (inline) {
        return (
          <code className="bg-muted px-1 sm:px-1.5 py-0.5 rounded text-xs font-mono text-primary">
            {children}
          </code>
        )
      }
      return (
        <SyntaxHighlighter
          style={tomorrow}
          language={className?.replace("language-", "") || "text"}
          customStyle={{
            margin: "1rem 0",
            borderRadius: "0.5rem",
            overflow: "hidden",
            fontSize: "0.75rem",
          }}
        >
          {String(children).replace(/\n$/, "")}
        </SyntaxHighlighter>
      )
    },
    table: ({ children }: { children?: React.ReactNode }) => (
      <div className="overflow-x-auto mb-4">
        <table className="w-full border-collapse border border-border/50 bg-card text-xs sm:text-sm">
          {children}
        </table>
      </div>
    ),
    th: ({ children }: { children?: React.ReactNode }) => (
      <th className="border border-border/30 px-2 sm:px-3 py-1.5 sm:py-2 text-left font-semibold text-foreground/90 bg-muted/50">
        {children}
      </th>
    ),
    td: ({ children }: { children?: React.ReactNode }) => (
      <td className="border border-border/30 px-2 sm:px-3 py-1.5 sm:py-2 text-xs sm:text-sm text-foreground/80">
        {children}
      </td>
    ),
    blockquote: ({ children }: { children?: React.ReactNode }) => (
      <blockquote className="border-l-4 border-primary/20 bg-primary/5 pl-3 sm:pl-4 py-2 my-3 sm:my-4 italic text-foreground/70">
        {children}
      </blockquote>
    ),
  }), [])

  // Render text content using ReactMarkdown for modern, safe rendering
  const renderContent = useCallback((item: ProcessedItem) => {
    const text = item.text
    
    // Debug logging
    console.log('Rendering item:', { 
      id: item.id, 
      category: item.category, 
      text: text,
      textLength: text?.length,
      hasText: !!text,
      trimmedLength: text?.trim()?.length 
    })
    
    // Handle missing or empty text
    if (!text || text.trim() === "") {
      return (
        <div id={`item-${item.id}`} className="transition-all duration-150 rounded-md px-2 py-1">
          <p className="text-muted-foreground italic text-xs sm:text-sm">
            [{item.category}] - {!text ? 'No text field' : 'Empty text'}
          </p>
        </div>
      )
    }

    return (
      <div id={`item-${item.id}`} className="transition-all duration-150 rounded-md prose prose-sm max-w-none">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          rehypePlugins={[rehypeRaw]}
          components={markdownComponents}
        >
          {text}
        </ReactMarkdown>
      </div>
    )
  }, [markdownComponents])

  return (
    <div className="w-full h-full border-l border-border bg-background/80 backdrop-blur-sm flex flex-col shadow-sm">
      {/* Header */}
      <div className="border-b border-border/50 px-3 sm:px-4 py-2.5 sm:py-3 flex items-center justify-between sticky top-0 bg-inherit z-10 flex-shrink-0">
        <h2 className="text-xs sm:text-sm font-semibold text-foreground tracking-tight">Parsed Content</h2>
        {parsedData && (
          <div className="flex items-center gap-1 sm:gap-1.5">
            <Button
              onClick={handleCopy}
              variant="ghost"
              size="sm"
              className="h-7 w-7 sm:h-8 sm:w-8 p-0"
              aria-label="Copy results"
              title="Copy all"
            >
              <Copy className="w-3 h-3 sm:w-3.5 sm:h-3.5" />
            </Button>
            <Button
              onClick={handleDownload}
              variant="ghost"
              size="sm"
              className="h-7 w-7 sm:h-8 sm:w-8 p-0"
              aria-label="Download as text"
              title="Download TXT"
            >
              <Download className="w-3 h-3 sm:w-3.5 sm:h-3.5" />
            </Button>
            <Button
              onClick={handleDownloadJson}
              variant="ghost"
              size="sm"
              className="h-7 w-7 sm:h-8 sm:w-8 p-0"
              aria-label="Download as JSON"
              title="Download JSON"
            >
              <FileJson className="w-3 h-3 sm:w-3.5 sm:h-3.5" />
            </Button>
          </div>
        )}
      </div>

      {/* Content Area */}
      <div className="flex-1 flex flex-col overflow-hidden relative">
        {error && (
          <div className="mx-3 sm:mx-4 mt-3 sm:mt-4 p-2.5 sm:p-3 bg-destructive/5 border border-destructive/20 rounded-lg shadow-sm">
            <p className="text-xs sm:text-sm font-medium text-destructive">Processing Error</p>
            <p className="text-xs text-destructive/70 mt-1">{error}</p>
          </div>
        )}
        {isProcessing ? (
          <div className="flex-1 flex items-center justify-center p-6 sm:p-8">
            <div className="text-center space-y-3">
              <div className="inline-flex items-center justify-center w-10 h-10 sm:w-12 sm:h-12 rounded-full bg-primary/10">
                <Loader2 className="w-5 h-5 sm:w-6 sm:h-6 text-primary animate-spin" />
              </div>
              <div>
                <p className="text-sm sm:text-base text-foreground font-medium">Analyzing Document</p>
                <p className="text-xs sm:text-sm text-muted-foreground truncate max-w-xs">{processingFile}</p>
              </div>
            </div>
          </div>
        ) : parsedData ? (
          <div ref={contentRef} className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-4 sm:space-y-6">
            {groupedByPage.length > 0 ? (
              groupedByPage.map(({ pageNum, items }) => (
                <section key={`page-${pageNum}`} className="space-y-2 sm:space-y-3">
                  <div className="flex items-center gap-2 mb-2 sm:mb-3 pt-1">
                    <div className="w-1 h-5 sm:h-6 bg-primary/20 rounded-full" />
                    <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                      Page {pageNum}
                    </span>
                  </div>
                  <div className="space-y-2 sm:space-y-2.5">
                    {items.map((item) => (
                      <div
                        key={item.id}
                        className={`group relative rounded-lg border border-transparent transition-all duration-200 hover:border-border/30 ${
                          hoveredBbox === item.id
                            ? "bg-primary/5 border-primary/20 ring-1 ring-primary/10 shadow-sm -mx-1 sm:-mx-1.5 px-1 sm:px-1.5 py-1 sm:py-1.5"
                            : "px-1.5 sm:px-2 py-1 sm:py-1.5"
                        }`}
                      >
                        {hoveredBbox === item.id && (
                          <div className="hidden sm:block absolute -top-8 left-2 bg-primary/90 text-primary-foreground text-xs px-2 py-1 rounded-md shadow-lg whitespace-nowrap z-20">
                            {item.category}
                          </div>
                        )}
                        <div className="text-xs sm:text-sm leading-relaxed">
                          {renderContent(item)}
                        </div>
                      </div>
                    ))}
                  </div>
                </section>
              ))
            ) : (
              <div className="flex items-center justify-center h-full">
                <div className="text-center space-y-2">
                  <div className="w-12 h-12 sm:w-16 sm:h-16 bg-muted rounded-xl flex items-center justify-center mx-auto">
                    <span className="text-2xl sm:text-3xl">📄</span>
                  </div>
                  <div>
                    <p className="text-sm sm:text-base text-foreground font-medium">No Content Detected</p>
                    <p className="text-xs sm:text-sm text-muted-foreground">Document processed, but no extractable text found.</p>
                  </div>
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="flex-1 flex items-center justify-center p-6 sm:p-8">
            {uploadedFile ? (
              <div className="w-full max-w-md">
                <SettingsPanel />
              </div>
            ) : (
              <div className="text-center space-y-2">
                <div className="w-12 h-12 sm:w-16 sm:h-16 bg-muted rounded-xl flex items-center justify-center mx-auto">
                  <span className="text-2xl sm:text-3xl">📄</span>
                </div>
                <div>
                  <p className="text-sm sm:text-base text-foreground font-medium">Ready to Parse</p>
                  <p className="text-xs sm:text-sm text-muted-foreground">Upload a document to begin extraction.</p>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Footer Actions */}
      {uploadedFile && !parsedData && !isProcessing && (
        <div className="border-t border-border/50 px-3 sm:px-4 py-2.5 sm:py-3 bg-inherit flex-shrink-0">
          <Button className="w-full h-9 sm:h-10 font-medium text-sm" onClick={handleRunOcr}>
            Start Parsing
          </Button>
        </div>
      )}
      {isProcessing && (
        <div className="border-t border-border/50 px-3 sm:px-4 py-2.5 sm:py-3 bg-inherit flex-shrink-0">
          <Button 
            variant="outline" 
            className="w-full h-9 sm:h-10 font-medium text-sm"
            onClick={handleCancelOcr}
          >
            Cancel Processing
          </Button>
        </div>
      )}
    </div>
  )
}
