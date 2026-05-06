import { useState, useEffect, useRef, useCallback } from 'react';
import { X, Play, Loader2, FileText, Layout, AlertTriangle, CheckCircle, BrainCircuit } from 'lucide-react';
import { agentService, type CadAnalysis, type LayoutAnalysis, type Defect, type AgentReport } from '../services/agentService';
import { CadAnalysisView, LayoutAnalysisView, DefectAnalysisView, ReportView, DefectDetailView, type DefectSeverityFilter } from './agent/ResultViews';

interface AgentPanelProps {
    fileId: string;
    query: string;
    runId?: number;
    severityFilter: DefectSeverityFilter;
    onSeverityFilterChange: (next: DefectSeverityFilter) => void;
    onDefectsChange?: (defects: Defect[]) => void;
    onClose: () => void;
    onDefectSelect?: (defect: Defect | null) => void;
}

type Tab = 'cad' | 'layout' | 'defects' | 'report';

interface LogEntry {
    level?: string;
    message?: string;
    timestamp?: string;
    context?: Record<string, unknown>;
}

export default function AgentPanel({ fileId, query, runId, severityFilter, onSeverityFilterChange, onDefectsChange, onClose, onDefectSelect }: AgentPanelProps) {
    const [activeTab, setActiveTab] = useState<Tab>('cad');
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [hasStarted, setHasStarted] = useState(false);
    const [statusMessage, setStatusMessage] = useState<string>("");
    const hasReceivedReportRef = useRef(false);
    const hasFinalizedRef = useRef(false);

    // Data States
    const [cadData, setCadData] = useState<CadAnalysis | null>(null);
    const [layoutData, setLayoutData] = useState<LayoutAnalysis[]>([]);
    const [defectsData, setDefectsData] = useState<Defect[]>([]);
    const [reportData, setReportData] = useState<AgentReport[]>([]);
    const [selectedDefect, setSelectedDefect] = useState<Defect | null>(null);
    const [progressEntries, setProgressEntries] = useState<LogEntry[]>([]);

    const eventSourceRef = useRef<EventSource | null>(null);
    const lastRunIdRef = useRef<number | undefined>(undefined);
    const queryRef = useRef(query);

    useEffect(() => {
        queryRef.current = query;
    }, [query]);

    const startStream = useCallback(() => {
        if (!fileId) return;

        if (eventSourceRef.current) {
            eventSourceRef.current.close();
            eventSourceRef.current = null;
        }

        const streamUrl = agentService.getStreamUrl(fileId);
        const evtSource = new EventSource(streamUrl);
        eventSourceRef.current = evtSource;

        const finalizeFromServer = async () => {
            if (hasFinalizedRef.current) return;
            hasFinalizedRef.current = true;
            try {
                const statusData = await agentService.getStatus(fileId);
                if (statusData?.status === 'completed') {
                    setHasStarted(true);
                    setIsAnalyzing(false);
                    setStatusMessage("Analysis complete");

                    const [cadRes, layoutRes, defectsRes, reportRes] = await Promise.allSettled([
                        agentService.getCadAnalysis(fileId),
                        agentService.getLayoutAnalysis(fileId),
                        agentService.getDefects(fileId),
                        agentService.getReport(fileId)
                    ]);

                    if (cadRes.status === 'fulfilled') setCadData(cadRes.value);
                    if (layoutRes.status === 'fulfilled') setLayoutData(layoutRes.value);
                    if (defectsRes.status === 'fulfilled') setDefectsData(defectsRes.value);
                    if (reportRes.status === 'fulfilled') setReportData(reportRes.value);
                } else if (statusData?.status === 'failed') {
                    setHasStarted(true);
                    setIsAnalyzing(false);
                    setStatusMessage(`Analysis failed: ${statusData.details?.error || 'Unknown error'}`);
                } else {
                    hasFinalizedRef.current = false;
                    setStatusMessage(statusData?.details?.message || "Reconnecting...");
                }
            } catch {
                hasFinalizedRef.current = false;
            } finally {
                if (hasFinalizedRef.current && eventSourceRef.current) {
                    eventSourceRef.current.close();
                    eventSourceRef.current = null;
                }
            }
        };

        evtSource.onmessage = (event) => {
            try {
                const parsed = JSON.parse(event.data);
                const type = parsed.type;
                const data = parsed.data;

                if (type === 'cad_analysis') {
                    setCadData(data);
                    setStatusMessage("CAD Analysis complete");
                } else if (type === 'layout_analysis') {
                    setLayoutData(prev => [...prev, data]);
                    setStatusMessage(`Analyzed layout ${data.layout_id}`);
                } else if (type === 'defect_found') {
                    const newDefects = data.defects || [];
                    setDefectsData(prev => [...prev, ...newDefects]);
                    setStatusMessage(`Found ${newDefects.length} potential issues`);
                } else if (type === 'report') {
                    setReportData(prev => {
                        if (!data?.layout_id) return prev;
                        const next = prev.filter(r => r.layout_id !== data.layout_id);
                        next.push(data);
                        return next;
                    });
                    hasReceivedReportRef.current = true;
                    setStatusMessage("Report generated");
                } else if (type === 'log') {
                    if (data?.message) {
                        setStatusMessage(data.message);
                    }
                    setProgressEntries(prev => [...prev, data].slice(-6));
                } else if (type === 'status') {
                    if (data?.message) {
                        setStatusMessage(data.message);
                        setProgressEntries(prev => [...prev, { message: data.message, level: 'status', timestamp: data.timestamp }].slice(-6));
                    }
                } else if (type === 'error') {
                    setStatusMessage(`Error: ${data.message}`);
                    setIsAnalyzing(false);
                    evtSource.close();
                }
            } catch (e) {
                console.error("Error parsing SSE event:", e);
            }
        };

        evtSource.onerror = (err) => {
            console.error("EventSource failed:", err);
            if (evtSource.readyState === EventSource.CLOSED || hasReceivedReportRef.current) {
                void finalizeFromServer();
                return;
            }
        };
    }, [fileId]);

    const handleStartAnalysis = useCallback(async () => {
        if (!fileId) return;

        setIsAnalyzing(true);
        setHasStarted(true);
        setStatusMessage("Starting analysis...");

        setCadData(null);
        setLayoutData([]);
        setDefectsData([]);
        setReportData([]);
        setSelectedDefect(null);
        setProgressEntries([]);
        hasReceivedReportRef.current = false;
        hasFinalizedRef.current = false;
        if (onDefectSelect) onDefectSelect(null);

        try {
            await agentService.startAnalysis(fileId, queryRef.current);
            startStream();
        } catch (error) {
            console.error("Failed to start analysis:", error);
            setStatusMessage("Failed to start analysis");
            setIsAnalyzing(false);
        }
    }, [fileId, onDefectSelect, startStream]);

    useEffect(() => {
        if (!runId) return;
        if (!fileId) return;
        if (lastRunIdRef.current === runId) return;
        lastRunIdRef.current = runId;
        void handleStartAnalysis();
    }, [runId, fileId, handleStartAnalysis]);

    // Clean up EventSource on unmount
    useEffect(() => {
        return () => {
            if (eventSourceRef.current) {
                eventSourceRef.current.close();
            }
        };
    }, []);

    // Fetch existing data on mount if available
    useEffect(() => {
        if (!fileId) return;
        const checkStatus = async () => {
            try {
                const statusData = await agentService.getStatus(fileId);

                if (statusData.status === 'completed') {
                    setHasStarted(true);
                    setIsAnalyzing(false);
                    setStatusMessage("Analysis complete");

                    // Fetch all results
                    const [cadRes, layoutRes, defectsRes, reportRes] = await Promise.allSettled([
                        agentService.getCadAnalysis(fileId),
                        agentService.getLayoutAnalysis(fileId),
                        agentService.getDefects(fileId),
                        agentService.getReport(fileId)
                    ]);

                    if (cadRes.status === 'fulfilled') setCadData(cadRes.value);
                    if (layoutRes.status === 'fulfilled') setLayoutData(layoutRes.value);
                    if (defectsRes.status === 'fulfilled') setDefectsData(defectsRes.value);
                    if (reportRes.status === 'fulfilled') setReportData(reportRes.value);

                } else if (statusData.status === 'processing') {
                    setHasStarted(true);
                    setIsAnalyzing(true);
                    setStatusMessage(statusData.details?.message || "Resuming analysis stream...");
                    
                    const progress = statusData.details?.progress;
                    
                    // Fetch current partial data based on progress flags
                    if (progress?.cad_analysis) {
                         agentService.getCadAnalysis(fileId).then(d => d.summary !== "Not available" && setCadData(d)).catch(() => {});
                    }
                    
                    if (progress?.layouts_analyzed && progress.layouts_analyzed.length > 0) {
                        agentService.getLayoutAnalysis(fileId).then(d => setLayoutData(d)).catch(() => {});
                    }
                    
                    if (progress?.defects_count && progress.defects_count > 0) {
                        agentService.getDefects(fileId).then(d => setDefectsData(d)).catch(() => {});
                    }

                    if (progress?.report_generated) {
                         agentService.getReport(fileId).then(d => setReportData(d)).catch(() => {});
                    }
                    
                    startStream();
                } else if (statusData.status === 'failed') {
                    setHasStarted(true);
                    setIsAnalyzing(false);
                    setStatusMessage(`Analysis failed: ${statusData.details?.error || 'Unknown error'}`);
                }
            } catch {
                // Status 404 means not started
                setHasStarted(false);
            }
        };

        checkStatus();
    }, [fileId, startStream]);

    useEffect(() => {
        if (!onDefectsChange) return;
        onDefectsChange(defectsData);
    }, [defectsData, onDefectsChange]);

    const handleDefectClick = (defect: Defect) => {
        setSelectedDefect(defect);
        if (onDefectSelect) {
            onDefectSelect(defect);
        }
    };

    const handleBackToDefects = () => {
        setSelectedDefect(null);
        if (onDefectSelect) {
            onDefectSelect(null);
        }
    };

    const handleExport = async () => {
        if (!fileId) return;
        try {
            const { blob, filename } = await agentService.exportResults(fileId);
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        } catch (error) {
            console.error("Failed to export results:", error);
            setStatusMessage("Failed to export results");
        }
    };

    const handleTabChange = (tab: Tab) => {
        if (tab !== 'defects' && selectedDefect) {
            setSelectedDefect(null);
            if (onDefectSelect) onDefectSelect(null);
        }
        setActiveTab(tab);
    };

    return (
        <div className="w-[500px] h-full bg-[#1a1d21] border-l border-white/10 flex flex-col shadow-2xl z-20 transition-all duration-300">
            {/* Header */}
            <div className="p-4 border-b border-white/10 bg-[#16181b] flex items-center justify-between">
                <h2 className="text-lg font-bold text-white flex items-center gap-2">
                    <BrainCircuit className="w-5 h-5 text-purple-500" />
                    Agent Analysis
                </h2>
                <button
                    onClick={onClose}
                    className="p-1.5 hover:bg-white/10 rounded-lg text-neutral-400 hover:text-white transition-colors"
                >
                    <X className="w-5 h-5" />
                </button>
            </div>

            {/* Main Content */}
            <div className="flex-1 flex flex-col overflow-hidden relative">

                {!hasStarted ? (
                    <div className="flex-1 flex flex-col items-center justify-center p-8 text-center space-y-6">
                        <div className="w-20 h-20 bg-white/5 rounded-full flex items-center justify-center relative">
                            <div className="absolute inset-0 bg-purple-500/20 rounded-full animate-pulse"></div>
                            <BrainCircuit className="w-10 h-10 text-purple-400" />
                        </div>
                        <div className="space-y-2">
                            <h3 className="text-lg font-medium text-white">Ready to Analyze</h3>
                            <p className="text-sm text-neutral-400 max-w-xs mx-auto">
                                Run the AI agent to detect design defects, analyze layout structure, and verify CAD standards.
                            </p>
                        </div>
                        <button
                            onClick={handleStartAnalysis}
                            className="px-6 py-3 bg-purple-600 hover:bg-purple-500 text-white rounded-lg font-medium shadow-lg shadow-purple-900/20 flex items-center gap-2 transition-all hover:scale-105"
                        >
                            <Play className="w-4 h-4 fill-white" />
                            Start Analysis
                        </button>
                    </div>
                ) : (
                    // Results Interface
                    <div className="flex flex-col h-full">
                        {/* Status Bar */}
                        {hasStarted && (
                            <div className="px-4 py-3 bg-purple-500/10 border-b border-purple-500/20">
                                <div className="flex items-center gap-3">
                                    {isAnalyzing && <Loader2 className="w-4 h-4 text-purple-400 animate-spin" />}
                                    <span className="text-xs font-medium text-purple-200">{statusMessage}</span>
                                </div>
                                {progressEntries.length > 0 && (
                                    <div className="mt-2 space-y-1">
                                        {progressEntries.map((entry, index) => (
                                            <div key={`${entry.timestamp ?? "no-ts"}-${index}`} className="text-[11px] text-neutral-300">
                                                {entry.message ?? ""}
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>
                        )}

                        {/* Tabs */}
                        <div className="flex items-center px-4 pt-4 border-b border-white/5 gap-6">
                            <button
                                onClick={() => handleTabChange('cad')}
                                className={`pb-3 text-sm font-medium transition-colors relative ${activeTab === 'cad' ? 'text-white' : 'text-neutral-500 hover:text-neutral-300'
                                    }`}
                            >
                                <div className="flex items-center gap-2">
                                    <FileText className="w-4 h-4" />
                                    CAD
                                </div>
                                {activeTab === 'cad' && <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-purple-500 rounded-t-full" />}
                            </button>
                            <button
                                onClick={() => handleTabChange('layout')}
                                className={`pb-3 text-sm font-medium transition-colors relative ${activeTab === 'layout' ? 'text-white' : 'text-neutral-500 hover:text-neutral-300'
                                    }`}
                            >
                                <div className="flex items-center gap-2">
                                    <Layout className="w-4 h-4" />
                                    Layout <span className="text-[10px] bg-white/10 px-1.5 rounded-full text-neutral-400">{layoutData.length}</span>
                                </div>
                                {activeTab === 'layout' && <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-purple-500 rounded-t-full" />}
                            </button>
                            <button
                                onClick={() => handleTabChange('defects')}
                                className={`pb-3 text-sm font-medium transition-colors relative ${activeTab === 'defects' ? 'text-white' : 'text-neutral-500 hover:text-neutral-300'
                                    }`}
                            >
                                <div className="flex items-center gap-2">
                                    <AlertTriangle className="w-4 h-4" />
                                    Defects <span className={`text-[10px] px-1.5 rounded-full ${defectsData.length > 0 ? 'bg-red-500/20 text-red-400' : 'bg-white/10 text-neutral-400'}`}>{defectsData.length}</span>
                                </div>
                                {activeTab === 'defects' && <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-purple-500 rounded-t-full" />}
                            </button>
                            <button
                                onClick={() => handleTabChange('report')}
                                className={`pb-3 text-sm font-medium transition-colors relative ${activeTab === 'report' ? 'text-white' : 'text-neutral-500 hover:text-neutral-300'
                                    }`}
                            >
                                <div className="flex items-center gap-2">
                                    <CheckCircle className="w-4 h-4" />
                                    Report
                                </div>
                                {activeTab === 'report' && <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-purple-500 rounded-t-full" />}
                            </button>
                        </div>

                        {/* Tab Content */}
                        <div className="flex-1 overflow-y-auto p-4 custom-scrollbar">
                            {activeTab === 'cad' && (
                                <CadAnalysisView data={cadData!} />
                            )}
                            {activeTab === 'layout' && (
                                <LayoutAnalysisView data={layoutData} />
                            )}
                            {activeTab === 'defects' && (
                                selectedDefect ? (
                                    <DefectDetailView defect={selectedDefect} onBack={handleBackToDefects} />
                                ) : (
                                    <DefectAnalysisView 
                                        data={defectsData} 
                                        onSelect={handleDefectClick} 
                                        filter={severityFilter} 
                                        onFilterChange={onSeverityFilterChange} 
                                        onExport={handleExport}
                                    />
                                )
                            )}
                            {activeTab === 'report' && (
                                <ReportView data={reportData as unknown as AgentReport} />
                            )}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
