import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import type { CadAnalysis, LayoutAnalysis, Defect, AgentReport } from '../../services/agentService';
import { AlertTriangle, CheckCircle, Info, Layout, FileText, ArrowLeft, Brain, Terminal, Eye, ChevronDown, ChevronRight } from 'lucide-react';

// --- Helper Components ---

const Card = ({ title, icon: Icon, children, className = "" }: { title: string, icon?: React.ElementType<{ className?: string }>, children: React.ReactNode, className?: string }) => (
    <div className={`bg-white/5 border border-white/10 rounded-lg overflow-hidden ${className}`}>
        <div className="px-4 py-3 border-b border-white/5 bg-white/[0.02] flex items-center gap-2">
            {Icon && <Icon className="w-4 h-4 text-blue-400" />}
            <h3 className="text-sm font-medium text-neutral-200">{title}</h3>
        </div>
        <div className="p-4">
            {children}
        </div>
    </div>
);

const MarkdownRenderer = ({ content }: { content: string }) => {
    return (
        <ReactMarkdown
            components={{
                h1: ({ children }) => <h1 className="text-lg font-bold text-white mb-3 mt-4 first:mt-0">{children}</h1>,
                h2: ({ children }) => <h2 className="text-base font-bold text-white mb-2 mt-3 first:mt-0">{children}</h2>,
                h3: ({ children }) => <h3 className="text-sm font-bold text-neutral-200 mb-2 mt-3">{children}</h3>,
                p: ({ children }) => <p className="text-sm text-neutral-300 leading-relaxed mb-3 last:mb-0 whitespace-pre-wrap">{children}</p>,
                ul: ({ children }) => <ul className="list-disc list-inside space-y-1 mb-3 text-sm text-neutral-300">{children}</ul>,
                ol: ({ children }) => <ol className="list-decimal list-inside space-y-1 mb-3 text-sm text-neutral-300">{children}</ol>,
                li: ({ children }) => <li className="pl-1">{children}</li>,
                strong: ({ children }) => <strong className="text-white font-semibold">{children}</strong>,
                code: ({ children }) => <code className="bg-white/10 px-1 py-0.5 rounded text-xs font-mono text-neutral-200">{children}</code>,
                blockquote: ({ children }) => <blockquote className="border-l-2 border-white/20 pl-3 italic text-neutral-400 my-2">{children}</blockquote>
            }}
        >
            {content}
        </ReactMarkdown>
    );
};

const TrajectoryView = ({ trajectory }: { trajectory?: Record<string, unknown> }) => {
    const [expanded, setExpanded] = useState(false);

    if (!trajectory || Object.keys(trajectory).length === 0) return null;

    // Parse trajectory into steps
    const steps: { id: number, thought?: string, tool?: string, args?: unknown, observation?: unknown }[] = [];
    
    // Simple heuristic to group steps by index suffix (e.g. thought_0, tool_name_0)
    // We scan for keys like thought_X, tool_name_X, etc.
    const indices = new Set<number>();
    Object.keys(trajectory).forEach(key => {
        const match = key.match(/_(\d+)$/);
        if (match) indices.add(parseInt(match[1]));
    });
    
    Array.from(indices).sort((a, b) => a - b).forEach(i => {
        steps.push({
            id: i,
            thought: trajectory[`thought_${i}`] as string,
            tool: trajectory[`tool_name_${i}`] as string,
            args: trajectory[`tool_args_${i}`],
            observation: trajectory[`observation_${i}`]
        });
    });

    return (
        <div className="border border-white/10 rounded-lg overflow-hidden bg-white/[0.02]">
             <button 
                onClick={() => setExpanded(!expanded)}
                className="w-full px-4 py-2 bg-white/5 flex items-center justify-between hover:bg-white/10 transition-colors"
            >
                <div className="flex items-center gap-2">
                    <Brain className="w-4 h-4 text-purple-400" />
                    <span className="text-xs font-medium text-neutral-300 uppercase tracking-wider">Agent Thinking Process</span>
                    <span className="bg-white/10 text-neutral-400 text-[10px] px-1.5 rounded-full">{steps.length} steps</span>
                </div>
                {expanded ? <ChevronDown className="w-4 h-4 text-neutral-500" /> : <ChevronRight className="w-4 h-4 text-neutral-500" />}
            </button>
            
            {expanded && (
                <div className="p-4 space-y-6 bg-black/20">
                    {steps.map((step) => (
                        <div key={step.id} className="relative pl-6 border-l border-white/10 last:border-0 pb-6 last:pb-0">
                             <div className="absolute -left-1.5 top-0 w-3 h-3 rounded-full bg-white/10 border border-white/20" />
                             
                             {/* Thought */}
                             {step.thought && (
                                 <div className="mb-3">
                                     <div className="flex items-center gap-2 mb-1">
                                         <span className="text-[10px] font-bold text-purple-400 uppercase">Thought</span>
                                     </div>
                                     <p className="text-sm text-neutral-300 leading-relaxed bg-purple-500/5 p-3 rounded-md border border-purple-500/10">
                                         {step.thought}
                                     </p>
                                 </div>
                             )}
                             
                             {/* Tool Use */}
                             {step.tool && (
                                 <div className="mb-3">
                                     <div className="flex items-center gap-2 mb-1">
                                         <Terminal className="w-3 h-3 text-cyan-400" />
                                         <span className="text-[10px] font-bold text-cyan-400 uppercase">Action: {step.tool}</span>
                                     </div>
                                    {step.args != null && (
                                        <div className="bg-cyan-950/30 border border-cyan-900/50 rounded-md p-2 overflow-x-auto">
                                            <pre className="text-[10px] text-cyan-200 font-mono">
                                                {JSON.stringify(step.args, null, 2)}
                                            </pre>
                                        </div>
                                    )}
                                 </div>
                             )}
                             
                             {/* Observation */}
                            {step.observation != null && (
                                <div>
                                    <div className="flex items-center gap-2 mb-1">
                                        <Eye className="w-3 h-3 text-emerald-400" />
                                        <span className="text-[10px] font-bold text-emerald-400 uppercase">Observation</span>
                                    </div>
                                     <div className="bg-emerald-950/20 border border-emerald-900/30 rounded-md p-2 max-h-40 overflow-y-auto custom-scrollbar">
                                         <pre className="text-[10px] text-emerald-200/80 font-mono whitespace-pre-wrap">
                                             {typeof step.observation === 'string' 
                                                ? step.observation 
                                                : JSON.stringify(step.observation, null, 2)}
                                         </pre>
                                     </div>
                                 </div>
                             )}
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};

// --- View Components ---

export const CadAnalysisView = ({ data }: { data: CadAnalysis }) => {
    if (!data) return <div className="text-neutral-500 text-sm">No CAD analysis available.</div>;

    return (
        <div className="space-y-4">
            <Card title="Detailed Analysis" icon={FileText}>
                <MarkdownRenderer content={data.response || data.summary || ""} />
            </Card>

            {data.reasoning && (
                <Card title="AI Reasoning" icon={Info}>
                    <p className="text-sm text-neutral-400 italic leading-relaxed">{data.reasoning}</p>
                </Card>
            )}
            
            {data.trajectory && <TrajectoryView trajectory={data.trajectory} />}

            {data.technical_details && Object.keys(data.technical_details).length > 0 && (
                <Card title="Technical Details" icon={FileText}>
                    <div className="space-y-2">
                        {Object.entries(data.technical_details).map(([key, value]) => (
                            <div key={key} className="flex justify-between items-center py-1 border-b border-white/5 last:border-0">
                                <span className="text-xs text-neutral-500 capitalize">{key.replace(/_/g, ' ')}</span>
                                <span className="text-xs font-mono text-neutral-300">{String(value)}</span>
                            </div>
                        ))}
                    </div>
                </Card>
            )}
        </div>
    );
};

export const LayoutAnalysisView = ({ data }: { data: LayoutAnalysis[] }) => {
    if (!data || data.length === 0) return <div className="text-neutral-500 text-sm">No layout analysis available.</div>;

    return (
        <div className="space-y-4">
            {data.map((item, index) => (
                <Card key={index} title={`Layout ${item.layout_id}`} icon={Layout}>
                    <div className="space-y-4">
                        <MarkdownRenderer content={item.response} />
                        
                        {item.reasoning && (
                             <div className="bg-white/5 p-3 rounded-md border border-white/5">
                                <h4 className="text-[10px] font-bold text-neutral-500 uppercase tracking-wider mb-1">Reasoning</h4>
                                <p className="text-xs text-neutral-400 italic">{item.reasoning}</p>
                            </div>
                        )}
                        
                        {item.trajectory && <TrajectoryView trajectory={item.trajectory} />}
                    </div>
                </Card>
            ))}
        </div>
    );
};

export type DefectSeverityFilter = 'all' | 'high' | 'medium' | 'low';

export const DefectAnalysisView = ({
    data,
    onSelect,
    filter,
    onFilterChange,
    onExport
}: {
    data: Defect[];
    onSelect: (defect: Defect) => void;
    filter: DefectSeverityFilter;
    onFilterChange: (next: DefectSeverityFilter) => void;
    onExport?: () => void;
}) => {

    if (!data || data.length === 0) return <div className="text-neutral-500 text-sm text-center py-8">No defects detected.</div>;

    // Calculate Stats
    const stats = data.reduce((acc, defect) => {
        const severity = String(defect.severity).toLowerCase();
        if (severity === 'high') acc.high++;
        else if (severity === 'medium') acc.medium++;
        else acc.low++;
        return acc;
    }, { high: 0, medium: 0, low: 0 });

    const total = data.length;

    // Donut Chart Logic
    const radius = 40;
    const circumference = 2 * Math.PI * radius;
    const center = 50;

    const highDash = (stats.high / total) * circumference;
    const mediumDash = (stats.medium / total) * circumference;
    const lowDash = (stats.low / total) * circumference;

    // Offsets
    const highOffset = 0;
    const mediumOffset = -highDash;
    const lowOffset = -(highDash + mediumDash);

    // Filter Data
    const filteredData = data.filter(d => {
        if (filter === 'all') return true;
        return String(d.severity).toLowerCase() === filter;
    });

    const handleFilter = (type: 'high' | 'medium' | 'low') => {
        if (filter === type) onFilterChange('all');
        else onFilterChange(type);
    };

    const severityTextClass = (severity: unknown) => {
        const s = String(severity).toLowerCase();
        if (s === 'high') return 'text-red-300';
        if (s === 'medium') return 'text-yellow-300';
        return 'text-blue-300';
    };

    const listHeaderTextClass =
        filter === 'high'
            ? 'text-red-400'
            : filter === 'medium'
                ? 'text-yellow-400'
                : filter === 'low'
                    ? 'text-blue-400'
                    : 'text-neutral-500';

    const rowBgClass =
        filter === 'high'
            ? 'bg-red-500/5 hover:bg-red-500/10'
            : filter === 'medium'
                ? 'bg-yellow-500/5 hover:bg-yellow-500/10'
                : filter === 'low'
                    ? 'bg-blue-500/5 hover:bg-blue-500/10'
                    : 'hover:bg-white/5';

    return (
        <div className="space-y-6 bg-white/5 border border-white/10 rounded-xl p-4">
            {/* Stats Cards Row */}
            <div className="grid grid-cols-3 gap-3">
                <button 
                    onClick={() => handleFilter('high')}
                    className={`bg-red-500/10 border rounded-lg p-3 flex flex-col items-start justify-center text-left transition-all
                        ${filter === 'high' ? 'border-red-500 bg-red-500/20' : 'border-red-500/20 hover:bg-red-500/15'}
                    `}
                >
                    <span className="text-2xl font-bold !text-red-500 leading-none mb-1">{stats.high}</span>
                    <span className="text-[10px] !text-red-500 font-bold uppercase tracking-wider">High</span>
                </button>
                <button 
                    onClick={() => handleFilter('medium')}
                    className={`bg-yellow-500/10 border rounded-lg p-3 flex flex-col items-start justify-center text-left transition-all
                        ${filter === 'medium' ? 'border-yellow-500 bg-yellow-500/20' : 'border-yellow-500/20 hover:bg-yellow-500/15'}
                    `}
                >
                    <span className="text-2xl font-bold !text-yellow-500 leading-none mb-1">{stats.medium}</span>
                    <span className="text-[10px] !text-yellow-500 font-bold uppercase tracking-wider">Medium</span>
                </button>
                <button 
                    onClick={() => handleFilter('low')}
                    className={`bg-blue-500/10 border rounded-lg p-3 flex flex-col items-start justify-center text-left transition-all
                        ${filter === 'low' ? 'border-blue-500 bg-blue-500/20' : 'border-blue-500/20 hover:bg-blue-500/15'}
                    `}
                >
                    <span className="text-2xl font-bold !text-blue-500 leading-none mb-1">{stats.low}</span>
                    <span className="text-[10px] !text-blue-500 font-bold uppercase tracking-wider">Low</span>
                </button>
            </div>

            {/* Graphic Summary Section */}
            <div>
                <h3 className="text-xs font-bold text-neutral-500 uppercase tracking-wider mb-4 px-1">Graphic Summary</h3>
                <div className="flex items-center justify-center py-2">
                    <div className="relative w-32 h-32">
                         <svg className="w-full h-full transform -rotate-90" viewBox="0 0 100 100">
                            {/* Background Circle */}
                            <circle cx={center} cy={center} r={radius} fill="none" stroke="#2a2d33" strokeWidth="10" />
                            
                            {/* Critical Segment (Red) */}
                            {stats.high > 0 && (
                                <circle 
                                    cx={center} cy={center} r={radius} 
                                    fill="none" stroke="#ef4444" strokeWidth={filter === 'high' ? 12 : 10}
                                    strokeDasharray={`${highDash} ${circumference}`} 
                                    strokeDashoffset={highOffset}
                                    strokeLinecap="round"
                                    className={`transition-all ${filter === 'all' || filter === 'high' ? 'opacity-100' : 'opacity-25'} cursor-pointer`}
                                    onClick={() => handleFilter('high')}
                                />
                            )}
                            
                            {/* Warning Segment (Yellow) */}
                            {stats.medium > 0 && (
                                <circle 
                                    cx={center} cy={center} r={radius} 
                                    fill="none" stroke="#eab308" strokeWidth={filter === 'medium' ? 12 : 10}
                                    strokeDasharray={`${mediumDash} ${circumference}`} 
                                    strokeDashoffset={mediumOffset}
                                    strokeLinecap="round"
                                    className={`transition-all ${filter === 'all' || filter === 'medium' ? 'opacity-100' : 'opacity-25'} cursor-pointer`}
                                    onClick={() => handleFilter('medium')}
                                />
                            )}
                            
                            {/* Suggestion Segment (Blue) */}
                            {stats.low > 0 && (
                                <circle 
                                    cx={center} cy={center} r={radius} 
                                    fill="none" stroke="#3b82f6" strokeWidth={filter === 'low' ? 12 : 10}
                                    strokeDasharray={`${lowDash} ${circumference}`} 
                                    strokeDashoffset={lowOffset}
                                    strokeLinecap="round"
                                    className={`transition-all ${filter === 'all' || filter === 'low' ? 'opacity-100' : 'opacity-25'} cursor-pointer`}
                                    onClick={() => handleFilter('low')}
                                />
                            )}
                        </svg>
                    </div>
                </div>
            </div>

            {/* Defect List */}
            <div className="space-y-4">
                <div className="flex items-center justify-between px-1">
                    <h3 className={`text-xs font-bold uppercase tracking-wider ${listHeaderTextClass}`}>Defect List</h3>
                    <button 
                        onClick={onExport}
                        className="flex items-center gap-1 text-[10px] text-neutral-400 hover:text-white transition-colors"
                        title="Export to Excel"
                    >
                        <FileText className="w-3 h-3" />
                        Export
                    </button>
                </div>
                
                {filteredData.map((defect, index) => (
                    <div
                        key={defect.id ?? index}
                        onClick={() => onSelect(defect)}
                        className={`group cursor-pointer py-2 px-2 -mx-2 rounded-lg transition-colors ${rowBgClass}`}
                    >
                        <div className="flex items-stretch gap-3">
                            <div
                                className={`w-1 rounded-full ${String(defect.severity).toLowerCase() === 'high'
                                    ? 'bg-red-500'
                                    : String(defect.severity).toLowerCase() === 'medium'
                                        ? 'bg-yellow-500'
                                        : 'bg-blue-500'
                                    }`}
                            />
                            <div className={`mt-1 flex-shrink-0 
                                ${String(defect.severity).toLowerCase() === 'high' ? 'text-red-500' :
                                String(defect.severity).toLowerCase() === 'medium' ? 'text-yellow-500' :
                                    'text-blue-500'
                                }`}>
                                {String(defect.severity).toLowerCase() === 'high' ? (
                                    <div className="w-4 h-4 rounded-full border border-red-500 flex items-center justify-center">
                                        <div className="w-2 h-2 bg-red-500 rounded-full" />
                                    </div>
                                ) : String(defect.severity).toLowerCase() === 'medium' ? (
                                    <AlertTriangle className="w-4 h-4" />
                                ) : (
                                    <Info className="w-4 h-4" />
                                )}
                            </div>
                            
                            <div className="flex-1 min-w-0 pb-3 border-b border-white/5 group-last:border-0">
                                <div className="flex items-center justify-between mb-1">
                                    <span className={`text-sm font-bold ${severityTextClass(defect.severity)}`}>
                                        {defect.type || `Defect #${defect.id}`}
                                    </span>
                                    {/* Icon for extra content if needed */}
                                    <FileText className="w-3 h-3 text-neutral-600" />
                                </div>

                                <p className="text-xs text-neutral-400 leading-relaxed line-clamp-2">
                                    {defect.description}
                                </p>
                                {(defect.layout_id !== undefined || defect.chunk_id !== undefined) && (
                                    <div className="mt-2 text-[10px] text-neutral-500 font-mono">
                                        {defect.layout_id !== undefined ? `layout ${defect.layout_id}` : ''}
                                        {defect.layout_id !== undefined && defect.chunk_id !== undefined ? ' • ' : ''}
                                        {defect.chunk_id !== undefined ? `chunk ${defect.chunk_id}` : ''}
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};

export const DefectDetailView = ({ defect, onBack }: { defect: Defect, onBack: () => void }) => {
    return (
        <div className="flex flex-col h-full">
            <div className="flex items-center gap-2 mb-6 border-b border-white/10 pb-4">
                <button 
                    onClick={onBack}
                    className="p-1.5 hover:bg-white/10 rounded-lg text-neutral-400 hover:text-white transition-colors"
                >
                    <ArrowLeft className="w-5 h-5" />
                </button>
                <h3 className="text-lg font-bold text-white">Defect Details</h3>
            </div>

            <div className="space-y-6 overflow-y-auto pr-2 custom-scrollbar flex-1">
                <section>
                    <h4 className="text-xs font-bold text-neutral-500 uppercase tracking-wider mb-2">Problem Description</h4>
                    <p className="text-sm text-neutral-200 leading-relaxed bg-white/5 p-4 rounded-lg border border-white/10">
                        {defect.description}
                    </p>
                </section>

                {defect.type && (
                     <section>
                        <h4 className="text-xs font-bold text-neutral-500 uppercase tracking-wider mb-2">Type / Category</h4>
                        <p className="text-sm text-neutral-300">
                            {defect.type}
                        </p>
                    </section>
                )}

                {defect.how_to_fix && (
                    <section>
                        <h4 className="text-xs font-bold text-neutral-500 uppercase tracking-wider mb-2">Suggested Fix</h4>
                        <p className="text-sm text-cyan-300 leading-relaxed bg-cyan-500/10 p-4 rounded-lg border border-cyan-500/20">
                            {defect.how_to_fix}
                        </p>
                    </section>
                )}

                <section>
                    <h4 className="text-xs font-bold text-neutral-500 uppercase tracking-wider mb-2">Metrics & Info</h4>
                    <div className="grid grid-cols-2 gap-4">
                         <div className="bg-white/5 p-3 rounded-lg border border-white/10">
                            <div className="text-[10px] text-neutral-500 mb-1">Severity</div>
                            <div className={`text-sm font-bold ${
                                String(defect.severity).toLowerCase() === 'high' ? 'text-red-400' :
                                String(defect.severity).toLowerCase() === 'medium' ? 'text-yellow-400' :
                                'text-blue-400'
                            }`}>
                                {defect.severity}
                            </div>
                        </div>
                        {defect.location && (
                             <div className="bg-white/5 p-3 rounded-lg border border-white/10">
                                <div className="text-[10px] text-neutral-500 mb-1">Location</div>
                                <div className="text-sm text-neutral-200 truncate" title={defect.location}>
                                    {defect.location}
                                </div>
                            </div>
                        )}
                         {defect.chunk_id !== undefined && (
                             <div className="bg-white/5 p-3 rounded-lg border border-white/10">
                                <div className="text-[10px] text-neutral-500 mb-1">Chunk ID</div>
                                <div className="text-sm font-mono text-neutral-200">
                                    {defect.chunk_id}
                                </div>
                            </div>
                        )}
                        {defect.layout_id !== undefined && (
                             <div className="bg-white/5 p-3 rounded-lg border border-white/10">
                                <div className="text-[10px] text-neutral-500 mb-1">Layout ID</div>
                                <div className="text-sm font-mono text-neutral-200">
                                    {defect.layout_id}
                                </div>
                            </div>
                        )}
                    </div>
                </section>
            </div>
        </div>
    );
};

export const ReportView = ({ data }: { data: AgentReport[] | AgentReport }) => {
    const normalized = Array.isArray(data) ? data : data ? [data] : [];
    if (normalized.length === 0) return <div className="text-neutral-500 text-sm">No report available.</div>;

    const sortedReports = [...normalized].sort((a, b) => {
        const aNum = Number(a.layout_id);
        const bNum = Number(b.layout_id);
        if (!Number.isNaN(aNum) && !Number.isNaN(bNum)) return aNum - bNum;
        return String(a.layout_id).localeCompare(String(b.layout_id));
    });

    return (
        <div className="space-y-6">
            {sortedReports.map((item, index) => (
                <div key={`${item.layout_id}-${index}`} className="space-y-6">
                    <Card title={`Report for Layout ${item.layout_id}`} icon={CheckCircle}>
                        <MarkdownRenderer content={item.report.drawing_summary} />
                    </Card>

                    {item.report.overall_recommendations && (
                        <Card title="Recommendations" icon={Info}>
                            <MarkdownRenderer content={item.report.overall_recommendations} />
                        </Card>
                    )}

                    {item.report.sources && item.report.sources.length > 0 && (
                        <div className="space-y-2">
                            <h3 className="text-xs font-bold uppercase text-neutral-500 tracking-wider px-1">Sources</h3>
                            <div className="flex flex-wrap gap-2">
                                {item.report.sources.map((source, i) => (
                                    <span key={`${item.layout_id}-source-${i}`} className="px-2 py-1 rounded bg-white/5 border border-white/10 text-[10px] text-neutral-400 font-mono">
                                        {source}
                                    </span>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            ))}
        </div>
    );
};
