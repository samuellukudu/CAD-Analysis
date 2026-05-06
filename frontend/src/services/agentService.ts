import { API_ENDPOINTS } from '../config/api';

export interface AgentQuery {
    query?: string;
    file_id?: string;
}

export interface AgentAnalysisResponse {
    message: string;
    file_id: string;
    stream_url: string;
    endpoints: {
        cad_analysis: string;
        layout_analysis: string;
        defects: string;
        report: string;
    };
}

export interface Defect {
    id: string | number;
    type?: string;
    description: string;
    location?: string;
    severity: 'low' | 'medium' | 'high' | 'Low' | 'Medium' | 'High';
    how_to_fix?: string;
    layout_id?: string | number;
    chunk_id?: string | number;
    cluster_id?: string | number;
}

export interface LayoutAnalysis {
    layout_id: number | string;
    response: string;
    reasoning?: string;
    trajectory?: Record<string, unknown>;
}

export interface CadAnalysis {
    response: string;
    reasoning?: string;
    trajectory?: Record<string, unknown>;
    summary?: string; // Keep for backward compatibility/types
    technical_details?: Record<string, unknown>; // Keep/optional
}

export interface AgentReportData {
    drawing_summary: string;
    defects?: Defect[];
    overall_recommendations?: string;
    sources?: string[];
}

export interface AgentReport {
    layout_id: string;
    report: AgentReportData;
}

export type AgentReportResponse = AgentReport | AgentReport[];

export interface AgentProgress {
    cad_analysis: boolean;
    layouts_analyzed: string[];
    defects_count: number;
    report_generated: boolean;
}

export interface AgentStatusDetails {
    query?: string;
    message?: string;
    error?: string;
    progress?: AgentProgress;
    transform?: Record<string, unknown>;
}

export interface AgentStatusResponse {
    id: string;
    status: 'processing' | 'completed' | 'failed';
    details?: AgentStatusDetails;
}

export const agentService = {
    /**
     * Start the AI agent analysis pipeline.
     */
    startAnalysis: async (fileId?: string, query: string = "detect design errors"): Promise<AgentAnalysisResponse> => {
        const response = await fetch(API_ENDPOINTS.AGENT.ANALYZE, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ file_id: fileId, query }),
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || 'Failed to start analysis');
        }

        return response.json();
    },

    /**
     * Get the processing status.
     */
    getStatus: async (fileId: string): Promise<AgentStatusResponse> => {
        const response = await fetch(API_ENDPOINTS.AGENT.GET_STATUS(fileId));
        if (!response.ok) {
            if (response.status === 404) throw new Error('Status not found');
            throw new Error('Failed to fetch status');
        }
        return response.json();
    },

    /**
     * Get the EventSource URL for streaming updates.
     */
    getStreamUrl: (fileId: string): string => {
        return API_ENDPOINTS.AGENT.STREAM(fileId);
    },

    /**
     * Fetch CAD analysis results.
     */
    getCadAnalysis: async (fileId: string): Promise<CadAnalysis> => {
        const response = await fetch(API_ENDPOINTS.AGENT.GET_CAD_ANALYSIS(fileId));
        if (!response.ok) {
            if (response.status === 404) return { response: "Not available", summary: "Not available", technical_details: {} };
            throw new Error('Failed to fetch CAD analysis');
        }
        return response.json();
    },

    /**
     * Fetch Layout analysis results.
     */
    getLayoutAnalysis: async (fileId: string): Promise<LayoutAnalysis[]> => {
        const response = await fetch(API_ENDPOINTS.AGENT.GET_LAYOUT_ANALYSIS(fileId));
        if (!response.ok) {
            if (response.status === 404) return [];
            throw new Error('Failed to fetch layout analysis');
        }
        return response.json();
    },

    /**
     * Fetch Defect analysis results.
     */
    getDefects: async (fileId: string): Promise<Defect[]> => {
        const response = await fetch(API_ENDPOINTS.AGENT.GET_DEFECTS(fileId));
        if (!response.ok) {
            if (response.status === 404) return [];
            throw new Error('Failed to fetch defects');
        }
        return response.json();
    },

    /**
     * Fetch final report.
     */
    getReport: async (fileId: string): Promise<AgentReport[]> => {
        const response = await fetch(API_ENDPOINTS.AGENT.GET_REPORT(fileId));
        if (!response.ok) {
            if (response.status === 404) return [];
            throw new Error('Failed to fetch report');
        }
        const payload: AgentReportResponse = await response.json();
        if (Array.isArray(payload)) return payload;
        if (!payload) return [];
        return [payload];
    },

    /**
     * Export analysis results as Excel file.
     */
    exportResults: async (fileId: string): Promise<{ blob: Blob; filename: string }> => {
        const response = await fetch(API_ENDPOINTS.AGENT.EXPORT_XLSX(fileId));
        if (!response.ok) {
            throw new Error('Failed to export results');
        }
        
        // Try to get filename from Content-Disposition header
        let filename = `${fileId}_results.xlsx`;
        const disposition = response.headers.get('Content-Disposition');
        if (disposition && disposition.indexOf('attachment') !== -1) {
            // Priority 1: Handle filename*=utf-8''encoded_name (RFC 6266)
            const filenameStarRegex = /filename\*=utf-8''([^;]+)/i;
            const matchesStar = filenameStarRegex.exec(disposition);
            
            if (matchesStar && matchesStar[1]) {
                try {
                    filename = decodeURIComponent(matchesStar[1]);
                } catch (e) {
                    console.warn('Failed to decode filename from Content-Disposition', e);
                }
            } else {
                // Priority 2: Handle filename="name" or filename=name
                const filenameRegex = /filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/;
                const matches = filenameRegex.exec(disposition);
                if (matches != null && matches[1]) { 
                    filename = matches[1].replace(/['"]/g, '');
                }
            }
        }
        
        return {
            blob: await response.blob(),
            filename
        };
    }
};
