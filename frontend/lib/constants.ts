// Type definitions
export interface BoundingBox {
  bbox: [number, number, number, number]
  category: string
  text: string
}

export interface PageResult {
  status: string
  data: BoundingBox[] | string // Can be array of boxes or markdown string
  error: string | null
}

export interface ParsedData {
  filename: string
  total_pages: number
  dpi: number
  model: string
  results_by_page: Record<string, PageResult>
  message?: string
  success?: boolean
}

// Color map for different content categories
export const CATEGORY_COLOR_MAP: Record<string, string> = {
  Text: "#00AA00",
  Title: "#FF0000",
  "Section-header": "#FF00FF",
  Picture: "#00FFFF",
  Table: "#CC9900",
  Formula: "#FFA500",
  Caption: "#FFC0CB",
  Footnote: "#A9A9A9",
  "List-item": "#228B22",
  "Page-header": "#87CEEB",
  "Page-footer": "#DDA0DD",
}

// Sample mock data for testing
export const SAMPLE_PARSED_DATA: ParsedData = {
  filename: "FORM-2.pdf",
  total_pages: 6,
  dpi: 200,
  model: "dotsocr",
  results_by_page: {
    page_0: {
      status: "success",
      data: [
        {
          bbox: [380, 358, 1327, 526],
          category: "Title",
          text: "FORM 2\n[Refer Rules 10, 14, 17 and 18]\nFORM OF APPLICATION FOR LEARNER'S LICENCE OR DRIVING LICENCE OR ADDITION OF A NEW CLASS OF VEHICLE OR RENEWAL OF DRIVING LICENCE OR CHANGE OF ADDRESS OR NAME"
        },
        {
          bbox: [353, 552, 394, 581],
          category: "Text",
          text: "To,"
        },
        {
          bbox: [353, 614, 633, 645],
          category: "Text",
          text: "The Licencing Authority"
        },
        {
          bbox: [351, 740, 1354, 1161],
          category: "Table",
          text: "<table><thead><tr><th colspan=\"2\">Services applying for</th></tr></thead></table>"
        }
      ],
      error: null
    },
    page_1: {
      status: "success",
      data: [
        {
          bbox: [349, 711, 1039, 741],
          category: "Section-header",
          text: "2. Personal details of the Applicant (in Capital Letters)"
        },
        {
          bbox: [350, 741, 1355, 1304],
          category: "Table",
          text: "<table><tr><td>Details of Aadhaar card</td></tr></table>"
        }
      ],
      error: null
    },
    page_2: {
      status: "success",
      data: [
        {
          bbox: [350, 350, 1356, 402],
          category: "Table",
          text: "<table><tr><td>Pin code</td><td></td><td></td></tr></table>"
        },
        {
          bbox: [348, 427, 1358, 489],
          category: "Section-header",
          text: "5. In case of request for Addition of a Class of Vehicle in Transport Category, please fill the following:"
        }
      ],
      error: null
    }
  }
}

// Utility function to convert hex color to rgba
export function hexToRgba(hex: string, alpha: number): string {
  const r = parseInt(hex.slice(1, 3), 16)
  const g = parseInt(hex.slice(3, 5), 16)
  const b = parseInt(hex.slice(5, 7), 16)
  return `rgba(${r}, ${g}, ${b}, ${alpha})`
}
