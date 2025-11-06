import { 
  ocrPdfOcrPost, 
  rootGet, 
  getAvailableModelsOcrModelsGet,
  submitOcrJobOcrSubmitPost,
  getJobStatusOcrStatusJobIdGet,
  getJobResultOcrResultJobIdGet
} from "@/api-client";
import type { OcrResponse, OcrJobSubmitResponse } from "@/api-client/types.gen";
import type { ParsedData, PageResult } from "./constants";

// Transform OcrResponse to ParsedData format
function transformOcrResponse(ocrResponse: OcrResponse): ParsedData {
  // Transform the results object to match PageResult structure
  // The backend returns results as a generic object, so we need to type-cast it
  const results_by_page = ocrResponse.results as unknown as Record<string, PageResult>;

  return {
    filename: ocrResponse.filename,
    total_pages: ocrResponse.total_pages,
    dpi: ocrResponse.dpi,
    model: ocrResponse.model,
    results_by_page,
    message: ocrResponse.message,
    success: ocrResponse.success,
  };
}

// Transform job result to ParsedData format
function transformJobResult(jobResult: any): ParsedData {
  const results_by_page = jobResult.results as unknown as Record<string, PageResult>;

  return {
    filename: jobResult.filename,
    total_pages: jobResult.total_pages,
    dpi: jobResult.dpi,
    model: jobResult.model,
    results_by_page,
    message: jobResult.message || "Job completed successfully",
    success: jobResult.success !== false,
  };
}

/**
 * Health check endpoint
 * @returns Promise with the API health status
 */
export async function checkHealth() {
  try {
    const response = await rootGet();
    return {
      success: true,
      data: response.data,
    };
  } catch (error) {
    console.error("Health check failed:", error);
    return {
      success: false,
      error: error instanceof Error ? error.message : "Unknown error",
    };
  }
}

/**
 * Upload and process a PDF file for OCR
 * @param file - The PDF file to process
 * @param options - Optional configuration for OCR processing
 * @returns Promise with the parsed OCR results
 */
export async function uploadPdfForOcr(
  file: File,
  options?: {
    dpi?: number;
    model?: string;
    annotate?: boolean;
    outputFolder?: string | null;
  }
) {
  try {
    // Validate file type
    if (!file.type.includes("pdf")) {
      throw new Error("Only PDF files are supported");
    }

    console.log("Uploading PDF for OCR:", file.name);
    console.log("API Base URL:", process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8003');

    const response = await ocrPdfOcrPost({
      body: {
        file,
      },
      query: {
        dpi: options?.dpi,
        model: options?.model,
        annotate: options?.annotate,
        output_folder: options?.outputFolder,
      },
    });

    console.log("Full OCR Response:", JSON.stringify(response, null, 2));

    // Check if response exists
    if (!response) {
      throw new Error("No response received from OCR service");
    }

    // Check if response data exists
    if (!response.data) {
      console.error("Response object:", response);
      throw new Error("No data in response from OCR service");
    }

    console.log("Response data received:", response.data);

    // Check if the backend reports success
    if (response.data.success === false) {
      throw new Error(response.data.message || "OCR processing failed on backend");
    }

    // Transform the OcrResponse to our internal format
    const transformedData = transformOcrResponse(response.data);

    console.log("Transformed data:", transformedData);

    return {
      success: true,
      data: transformedData,
    };
  } catch (error) {
    console.error("OCR processing failed:", error);
    // Log more details about the error
    if (error && typeof error === 'object') {
      console.error("Error details:", JSON.stringify(error, null, 2));
    }
    return {
      success: false,
      error: error instanceof Error ? error.message : "OCR processing failed",
      data: null,
    };
  }
}

/**
 * Process a PDF file with custom DPI settings
 * @param file - The PDF file to process
 * @param dpi - DPI setting for image conversion (default: 200)
 * @returns Promise with the parsed OCR results
 */
export async function processPdfWithDpi(file: File, dpi: number = 200) {
  return uploadPdfForOcr(file, { dpi });
}

/**
 * Batch process multiple PDF files
 * @param files - Array of PDF files to process
 * @param options - Optional configuration for OCR processing
 * @returns Promise with array of results for each file
 */
export async function batchProcessPdfs(
  files: File[],
  options?: {
    dpi?: number;
    model?: string;
    annotate?: boolean;
    outputFolder?: string | null;
  }
) {
  try {
    const results = await Promise.allSettled(
      files.map((file) => uploadPdfForOcr(file, options))
    );

    return results.map((result, index) => ({
      filename: files[index].name,
      status: result.status,
      ...(result.status === "fulfilled"
        ? { data: result.value }
        : { error: result.reason }),
    }));
  } catch (error) {
    console.error("Batch processing failed:", error);
    throw error;
  }
}

/**
 * Validate if the API is reachable
 * @returns Promise with boolean indicating if API is reachable
 */
export async function validateApiConnection(): Promise<boolean> {
  const result = await checkHealth();
  return result.success;
}

/**
 * Error handler for API calls
 * @param error - The error object
 * @returns Formatted error message
 */
export function handleApiError(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  if (typeof error === "string") {
    return error;
  }
  return "An unexpected error occurred";
}

/**
 * Parse error response from API
 * @param error - The error response
 * @returns User-friendly error message
 */
export function parseApiError(error: any): string {
  // Handle validation errors (422)
  if (error?.status === 422 && error?.data?.detail) {
    const details = error.data.detail as Array<{
      loc: Array<string | number>;
      msg: string;
      type: string;
    }>;
    return details.map((d) => `${d.loc.join(".")}: ${d.msg}`).join(", ");
  }

  // Handle other HTTP errors
  if (error?.status) {
    return `HTTP ${error.status}: ${error.statusText || "Unknown error"}`;
  }

  return handleApiError(error);
}

/**
 * Get available OCR models and their features
 * @returns Promise with information about available OCR models
 */
export async function getAvailableModels() {
  try {
    const response = await getAvailableModelsOcrModelsGet();
    
    if (!response) {
      throw new Error("No response received from models endpoint");
    }

    return {
      success: true,
      data: response.data,
    };
  } catch (error) {
    console.error("Failed to fetch available models:", error);
    return {
      success: false,
      error: error instanceof Error ? error.message : "Failed to fetch models",
    };
  }
}

/**
 * Process a PDF file with a specific OCR model
 * @param file - The PDF file to process
 * @param model - The OCR model to use (e.g., 'dotsocr', 'lightonocr')
 * @param options - Additional processing options
 * @returns Promise with the parsed OCR results
 */
export async function processPdfWithModel(
  file: File,
  model: string,
  options?: {
    dpi?: number;
    annotate?: boolean;
    outputFolder?: string | null;
  }
) {
  return uploadPdfForOcr(file, { model, ...options });
}

/**
 * Process a PDF file with annotations enabled
 * @param file - The PDF file to process
 * @param options - Additional processing options
 * @returns Promise with the parsed OCR results including annotated images
 */
export async function processPdfWithAnnotations(
  file: File,
  options?: {
    dpi?: number;
    model?: string;
    outputFolder?: string | null;
  }
) {
  return uploadPdfForOcr(file, { annotate: true, ...options });
}

/**
 * Process a PDF file using the DotsOCR model for structured layout
 * @param file - The PDF file to process
 * @param options - Additional processing options
 * @returns Promise with structured layout JSON including bbox and categories
 */
export async function processPdfWithDotsOcr(
  file: File,
  options?: {
    dpi?: number;
    annotate?: boolean;
    outputFolder?: string | null;
  }
) {
  return uploadPdfForOcr(file, { model: "dotsocr", ...options });
}

// ============================================================
// JOB QUEUE API (New Asynchronous Processing)
// ============================================================

/**
 * Submit a PDF for OCR processing using the job queue
 * @param file - The PDF file to process
 * @param options - Optional configuration for OCR processing
 * @returns Promise with job submission details including job_id
 */
export async function submitOcrJob(
  file: File,
  options?: {
    dpi?: number;
    model?: string;
  }
) {
  try {
    // Validate file type
    if (!file.type.includes("pdf")) {
      throw new Error("Only PDF files are supported");
    }

    console.log("Submitting OCR job:", file.name);
    console.log("API Base URL:", process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8003');

    const response = await submitOcrJobOcrSubmitPost({
      body: {
        file,
      },
      query: {
        dpi: options?.dpi,
        model: options?.model,
      },
    });

    console.log("Job submission response:", response);

    if (!response || !response.data) {
      throw new Error("No response received from OCR service");
    }

    return {
      success: true,
      data: response.data,
    };
  } catch (error) {
    console.error("Failed to submit OCR job:", error);
    return {
      success: false,
      error: error instanceof Error ? error.message : "Failed to submit OCR job",
      data: null,
    };
  }
}

/**
 * Check the status of an OCR job
 * @param jobId - The job ID to check
 * @returns Promise with job status information
 */
export async function checkJobStatus(jobId: string) {
  try {
    const response = await getJobStatusOcrStatusJobIdGet({
      path: { job_id: jobId },
    });

    if (!response || !response.data) {
      throw new Error("No response received from status endpoint");
    }

    return {
      success: true,
      data: response.data,
    };
  } catch (error) {
    console.error("Failed to check job status:", error);
    return {
      success: false,
      error: error instanceof Error ? error.message : "Failed to check job status",
      data: null,
    };
  }
}

/**
 * Get the result of a completed OCR job
 * @param jobId - The job ID to retrieve results for
 * @param annotate - Whether to include annotated images
 * @returns Promise with the parsed OCR results
 */
export async function getJobResult(
  jobId: string,
  annotate: boolean = false
) {
  try {
    const response = await getJobResultOcrResultJobIdGet({
      path: { job_id: jobId },
      query: { annotate },
    });

    console.log("Job result response:", response);

    if (!response) {
      throw new Error("No response received from result endpoint");
    }

    // Handle different response statuses
    if (response.status === 202) {
      return {
        success: false,
        status: "running",
        error: "Job is still being processed",
        data: null,
      };
    }

    if (response.status === 404) {
      return {
        success: false,
        status: "expired",
        error: "Job results have expired",
        data: null,
      };
    }

    if (!response.data) {
      throw new Error("No data in response from result endpoint");
    }

    // Transform the job result to our internal format
    const transformedData = transformJobResult(response.data);

    return {
      success: true,
      status: "completed",
      data: transformedData,
    };
  } catch (error) {
    console.error("Failed to get job result:", error);
    return {
      success: false,
      status: "error",
      error: error instanceof Error ? error.message : "Failed to get job result",
      data: null,
    };
  }
}

/**
 * Poll for job completion
 * @param jobId - The job ID to poll
 * @param onProgress - Callback for progress updates
 * @param maxAttempts - Maximum number of polling attempts (default: 150)
 * @param pollInterval - Interval between polls in ms (default: 2000)
 * @returns Promise with the parsed OCR results when complete
 */
export async function pollJobCompletion(
  jobId: string,
  onProgress?: (status: string, attempt: number) => void,
  maxAttempts: number = 150,
  pollInterval: number = 2000
): Promise<{ success: boolean; data: ParsedData | null; error?: string }> {
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    const statusResult = await checkJobStatus(jobId);

    if (!statusResult.success || !statusResult.data) {
      return {
        success: false,
        data: null,
        error: statusResult.error || "Failed to check job status",
      };
    }

    const status = statusResult.data.status;
    onProgress?.(status, attempt);

    if (status === "completed") {
      // Get the final result
      const result = await getJobResult(jobId);
      return result;
    }

    if (status === "failed" || status === "expired") {
      return {
        success: false,
        data: null,
        error: `Job ${status}: ${statusResult.data.message || "Unknown error"}`,
      };
    }

    // Wait before next poll
    await new Promise((resolve) => setTimeout(resolve, pollInterval));
  }

  return {
    success: false,
    data: null,
    error: "Polling timeout: Job did not complete within the expected time",
  };
}

/**
 * Stream job progress via SSE (Server-Sent Events)
 * @param jobId - The job ID to stream
 * @param onStatus - Callback for status updates
 * @param onResult - Callback when result is received
 * @param onError - Callback for errors
 * @returns EventSource instance for managing the connection
 */
export function streamJobProgress(
  jobId: string,
  callbacks: {
    onStatus?: (data: any) => void;
    onResult?: (data: ParsedData) => void;
    onError?: (error: string) => void;
    onTimeout?: (message: string) => void;
  }
): EventSource {
  const apiBase = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8003';
  const url = `${apiBase}/ocr/stream/${jobId}`;

  console.log("Starting SSE stream:", url);

  const eventSource = new EventSource(url);

  eventSource.addEventListener('status', (e) => {
    try {
      const data = JSON.parse(e.data);
      console.log("SSE status:", data);
      callbacks.onStatus?.(data);
    } catch (error) {
      console.error("Failed to parse status event:", error);
    }
  });

  eventSource.addEventListener('result', (e) => {
    try {
      const data = JSON.parse(e.data);
      console.log("SSE result:", data);
      
      // Transform to ParsedData format
      const transformedData = transformJobResult(data);
      callbacks.onResult?.(transformedData);
      
      eventSource.close();
    } catch (error) {
      console.error("Failed to parse result event:", error);
      callbacks.onError?.("Failed to parse result data");
    }
  });

  eventSource.addEventListener('error', (e: any) => {
    try {
      const data = e.data ? JSON.parse(e.data) : {};
      console.error("SSE error:", data);
      callbacks.onError?.(data.error || "Stream error");
      eventSource.close();
    } catch (error) {
      console.error("SSE connection error:", error);
      callbacks.onError?.("Stream connection error");
      eventSource.close();
    }
  });

  eventSource.addEventListener('timeout', (e) => {
    try {
      const data = JSON.parse(e.data);
      console.log("SSE timeout:", data);
      callbacks.onTimeout?.(data.message);
      eventSource.close();
    } catch (error) {
      console.error("Failed to parse timeout event:", error);
    }
  });

  eventSource.addEventListener('close', () => {
    console.log("SSE stream closed");
    eventSource.close();
  });

  eventSource.onerror = (error) => {
    console.error("SSE onerror:", error);
    callbacks.onError?.("Connection error");
    eventSource.close();
  };

  return eventSource;
}

/**
 * Process PDF using job queue with SSE streaming (Recommended)
 * @param file - The PDF file to process
 * @param options - Processing options
 * @param callbacks - Callbacks for progress updates
 * @returns Promise with the parsed OCR results
 */
export async function processPdfWithStreaming(
  file: File,
  options: {
    dpi?: number;
    model?: string;
  },
  callbacks: {
    onStatus?: (status: string, elapsed?: number) => void;
    onProgress?: (message: string) => void;
  }
): Promise<{ success: boolean; data: ParsedData | null; error?: string }> {
  return new Promise(async (resolve) => {
    try {
      // Submit the job
      const submitResult = await submitOcrJob(file, options);

      if (!submitResult.success || !submitResult.data) {
        resolve({
          success: false,
          data: null,
          error: submitResult.error || "Failed to submit job",
        });
        return;
      }

      const jobId = submitResult.data.job_id;
      callbacks.onProgress?.(`Job submitted: ${jobId}`);

      // Stream progress
      streamJobProgress(jobId, {
        onStatus: (data) => {
          callbacks.onStatus?.(data.status, data.elapsed);
          callbacks.onProgress?.(`Status: ${data.status}`);
        },
        onResult: (data) => {
          callbacks.onProgress?.("Processing complete!");
          resolve({ success: true, data });
        },
        onError: (error) => {
          resolve({ success: false, data: null, error });
        },
        onTimeout: (message) => {
          // Job is still running, poll for result
          callbacks.onProgress?.("Still processing, checking status...");
          pollJobCompletion(jobId, (status) => {
            callbacks.onStatus?.(status);
          }).then(resolve);
        },
      });
    } catch (error) {
      resolve({
        success: false,
        data: null,
        error: error instanceof Error ? error.message : "Unexpected error",
      });
    }
  });
}

/**
 * Process a PDF file using the LightonOCR model for clean markdown extraction
 * @param file - The PDF file to process
 * @param options - Additional processing options
 * @returns Promise with clean markdown text extraction
 */
export async function processPdfWithLightonOcr(
  file: File,
  options?: {
    dpi?: number;
    annotate?: boolean;
    outputFolder?: string | null;
  }
) {
  return uploadPdfForOcr(file, { model: 'lightonocr', ...options });
}

