/**
 * API Configuration
 * 
 * Centralized configuration for API endpoints.
 * You can override the API URL using environment variables:
 * - VITE_API_URL: Full API base URL (e.g., http://localhost:8000)
 * - VITE_API_PORT: API port (defaults to 8000)
 * - VITE_API_HOST: API host (defaults to localhost)
 */

// Get API URL from environment variable or use default
const getApiUrl = (): string => {
  // Check for full URL first (for production/deployment)
  if (import.meta.env.VITE_API_URL) {
    // Remove trailing slash if present
    return import.meta.env.VITE_API_URL.replace(/\/$/, '');
  }
  
  // Otherwise, construct from host and port
  // Use window.location.hostname to default to the same host as the frontend
  // This makes it work automatically on local network or server without hardcoding IP
  const host = import.meta.env.VITE_API_HOST || window.location.hostname;
  const port = import.meta.env.VITE_API_PORT || '8000';
  
  return `http://${host}:${port}`;
};

export const API_BASE_URL = getApiUrl();

// Log API configuration in development
if (import.meta.env.DEV) {
  console.log('[API Config] Base URL:', API_BASE_URL);
  console.log('[API Config] Upload endpoint:', `${API_BASE_URL}/api/dxf`);
}

// API endpoints
type ApiEndpoints = {
  UPLOAD_DXF: string;
  LIST_DXFS: string;
  GET_DXF: (id: string) => string;
  GET_STATUS: (id: string) => string;
  GET_MASKS: (id: string) => string;
  GET_IMAGE_CONTENT: (id: string) => string;
  GET_IMAGE_TRANSFORM: (id: string) => string;
  DOWNLOAD_DXF: (id: string) => string;
  DELETE_DXF: (id: string) => string;
  UPLOAD_PDF: string;
  LIST_PDFS: string;
  GET_PDF_STATUS: (id: string) => string;
  GET_PDF_CONTENT: (id: string) => string;
  SEARCH_PDFS: string;
  VIEW_PDF: string;
  DELETE_PDF: (id: string) => string;
  AGENT: {
    ANALYZE: string;
    GET_STATUS: (fileId: string) => string;
    STREAM: (fileId: string) => string;
    GET_CAD_ANALYSIS: (fileId: string) => string;
    GET_LAYOUT_ANALYSIS: (fileId: string) => string;
    GET_DEFECTS: (fileId: string) => string;
    GET_REPORT: (fileId: string) => string;
    EXPORT_XLSX: (fileId: string) => string;
  };
  CHAT: {
    STREAM: string;
  };
};

export const API_ENDPOINTS: ApiEndpoints = {
  UPLOAD_DXF: `${API_BASE_URL}/api/dxf`,
  LIST_DXFS: `${API_BASE_URL}/api/dxf`,
  GET_DXF: (id: string) => `${API_BASE_URL}/api/dxf/${id}`,
  GET_STATUS: (id: string) => `${API_BASE_URL}/api/dxf/${id}/status`,
  GET_MASKS: (id: string) => `${API_BASE_URL}/api/masks/${id}`,
  GET_IMAGE_CONTENT: (id: string) => `${API_BASE_URL}/api/images/${id}/content`,
  GET_IMAGE_TRANSFORM: (id: string) => `${API_BASE_URL}/api/images/${id}/transform`,
  DOWNLOAD_DXF: (id: string) => `${API_BASE_URL}/api/dxf/${id}/download`,
  DELETE_DXF: (id: string) => `${API_BASE_URL}/api/dxf/${id}`,

  // PDF Endpoints
  UPLOAD_PDF: `${API_BASE_URL}/api/pdfs`,
  LIST_PDFS: `${API_BASE_URL}/api/pdfs`,
  GET_PDF_STATUS: (id: string) => `${API_BASE_URL}/api/pdfs/${id}/status`,
  GET_PDF_CONTENT: (id: string) => `${API_BASE_URL}/api/pdfs/${id}/content`,
  SEARCH_PDFS: `${API_BASE_URL}/api/pdfs/search`,
  VIEW_PDF: `${API_BASE_URL}/api/pdfs/view`,
  DELETE_PDF: (id: string) => `${API_BASE_URL}/api/pdfs/${id}`,
  // Agent Endpoints
  AGENT: {
    ANALYZE: `${API_BASE_URL}/api/agent/analyze`,
    GET_STATUS: (fileId: string) => `${API_BASE_URL}/api/agent/${fileId}/status`,
    STREAM: (fileId: string) => `${API_BASE_URL}/api/agent/${fileId}/stream`,
    GET_CAD_ANALYSIS: (fileId: string) => `${API_BASE_URL}/api/agent/${fileId}/cad-analysis`,
    GET_LAYOUT_ANALYSIS: (fileId: string) => `${API_BASE_URL}/api/agent/${fileId}/layout-analysis`,
    GET_DEFECTS: (fileId: string) => `${API_BASE_URL}/api/agent/${fileId}/defects`,
    GET_REPORT: (fileId: string) => `${API_BASE_URL}/api/agent/${fileId}/report`,
    EXPORT_XLSX: (fileId: string) => `${API_BASE_URL}/api/agent/${fileId}/export-xlsx`,
  },
  CHAT: {
    STREAM: `${API_BASE_URL}/api/chat/stream`,
  },
};
