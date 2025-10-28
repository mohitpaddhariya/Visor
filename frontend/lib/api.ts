import { ocrPdfSaveOcrPost, rootGet } from "@/api-client";
import type { OcrResponse } from "@/api-client/types.gen";
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
    results_by_page,
    message: ocrResponse.message,
    success: ocrResponse.success,
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

    const response = await ocrPdfSaveOcrPost({
      body: {
        file,
      },
      query: {
        dpi: options?.dpi,
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
