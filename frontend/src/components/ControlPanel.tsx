import { useRef, useState } from 'react';
import type { ChangeEvent, ElementType, ReactNode } from 'react';
import { Layers, Box, Info, ChevronDown, ChevronRight, Hash, Activity, Share2, Map, Grid, Copy, Check, Network, PanelLeftClose, PanelLeftOpen, Image as ImageIcon, Eye, EyeOff, FileText, Trash2 } from 'lucide-react';
// getChunkColor will be used for dynamic mask coloring in the future
import type { FeatureLike } from '../utils/geoUtils';

interface ControlPanelProps {
  onUploadPdfs: (files: FileList) => void;
  onUploadDxfs: (files: FileList) => void;
  onSelectPdfId: (pdfId: string) => void;
  onSelectDxfId: (dxfId: string) => void;
  onDeletePdf?: (pdfId: string) => void;
  onDeleteDxf?: (dxfId: string) => void;
  pdfs: Array<{
    pdf_id: string;
    filename: string;
    pdf_source?: string | null;
    status?: string | null;
    status_details?: { step?: string; progress?: string; pages?: number };
  }>;
  dxfs: Array<{ 
    dxf_id: string; 
    filename: string; 
    source_filename?: string | null; 
    status?: string | null;
    status_details?: { step?: string; progress?: string; error?: string };
  }>;
  selectedPdfId: string | null;
  selectedDxfId: string | null;
  fileName: string | null;
  meta: {
    chunks: number;
    layouts: number;
  } | null;
  selectedFeature?: FeatureLike | null;
  viewMode?: 'overlay' | 'gallery' | 'graph' | 'raster';
  onViewModeChange?: (mode: 'overlay' | 'gallery' | 'graph' | 'raster') => void;
  showMasks?: boolean;
  onToggleMasks?: (show: boolean) => void;
  maskColorMode?: 'single' | 'distinct';
  onMaskColorModeChange?: (mode: 'single' | 'distinct') => void;
}

// --- UI Components ---

const Section = ({ title, icon: Icon, children, defaultOpen = true, isCollapsed = false }: { title: string, icon?: ElementType<{ className?: string }>, children: ReactNode, defaultOpen?: boolean, isCollapsed?: boolean }) => {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  if (isCollapsed) {
    return (
      <div className="py-4 border-b border-white/5 flex justify-center group relative">
        <div className="p-2 rounded-lg hover:bg-white/5 text-neutral-400 group-hover:text-white transition-colors cursor-help">
          {Icon && <Icon className="w-5 h-5" />}
        </div>
        {/* Tooltip on hover */}
        <div className="absolute left-full top-4 ml-2 px-2 py-1 bg-black/90 text-white text-xs rounded opacity-0 group-hover:opacity-100 whitespace-nowrap z-50 pointer-events-none transition-opacity border border-white/10">
          {title}
        </div>
      </div>
    );
  }

  return (
    <div className="border-b border-white/5 last:border-0">
      <button 
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between p-4 hover:bg-white/[0.02] transition-colors group"
      >
        <div className="flex items-center gap-2.5 text-sm font-medium text-neutral-300 group-hover:text-white transition-colors">
          {Icon && <Icon className="w-4 h-4 text-blue-500/80 group-hover:text-blue-500 transition-colors" />}
          {title}
        </div>
        {isOpen ? <ChevronDown className="w-4 h-4 text-neutral-600 group-hover:text-neutral-400" /> : <ChevronRight className="w-4 h-4 text-neutral-600 group-hover:text-neutral-400" />}
      </button>
      {isOpen && (
        <div className="px-4 pb-4 pt-0 animate-in fade-in slide-in-from-top-1 duration-200">
          {children}
        </div>
      )}
    </div>
  );
};

const DataRow = ({ label, value, mono = false }: { label: string, value: string | number, mono?: boolean }) => (
  <div className="flex justify-between items-center text-sm py-2 border-b border-white/[0.03] last:border-0">
    <span className="text-neutral-500 font-normal">{label}</span>
    <span className={`text-neutral-200 ${mono ? 'font-mono' : 'font-medium'}`}>{value}</span>
  </div>
);

const Tag = ({ children }: { children: ReactNode }) => (
  <span className="px-2.5 py-1 bg-white/[0.03] hover:bg-white/[0.08] text-neutral-300 hover:text-white rounded-full text-xs font-mono border border-white/10 transition-colors cursor-default select-all">
    {children}
  </span>
);

const CopyButton = ({ text }: { text: string }) => {
  const [copied, setCopied] = useState(false);
  
  const handleCopy = () => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <button 
      onClick={handleCopy}
      className="p-1 hover:bg-white/10 rounded transition-colors text-neutral-500 hover:text-neutral-300"
      title="Copy to clipboard"
    >
      {copied ? <Check className="w-3 h-3 text-green-500" /> : <Copy className="w-3 h-3" />}
    </button>
  );
};

export default function ControlPanel({ onUploadPdfs, onUploadDxfs, onSelectPdfId, onSelectDxfId, onDeletePdf, onDeleteDxf, pdfs, dxfs, selectedPdfId, selectedDxfId, fileName, meta, selectedFeature, viewMode = 'overlay', onViewModeChange, showMasks = true, onToggleMasks }: ControlPanelProps) {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const pdfInputRef = useRef<HTMLInputElement>(null);
  const dxfInputRef = useRef<HTMLInputElement>(null);

  const selectedProperties = selectedFeature?.properties ?? null;
  const geometryCoordinates =
    selectedFeature?.geometry && 'coordinates' in selectedFeature.geometry
      ? selectedFeature.geometry.coordinates
      : null;
  const geometryCoordinatesText = geometryCoordinates ? JSON.stringify(geometryCoordinates, null, 2) : '—';

  const getProperty = (key: string): unknown => {
    if (!selectedProperties) return undefined;
    return (selectedProperties as Record<string, unknown>)[key];
  };

  const layoutId = getProperty('layout_id');
  const chunkId = getProperty('chunk_id');
  const nodeId = getProperty('node_id');
  const graphFeatures = getProperty('features');
  const neighbors = getProperty('neighbors');

  const handlePdfChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      onUploadPdfs(e.target.files);
      e.target.value = '';
    }
  };

  const handleDxfChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      onUploadDxfs(e.target.files);
      e.target.value = '';
    }
  };

  return (
    <div className={`${isCollapsed ? 'w-16' : 'w-80'} h-full bg-[#1a1d21] border-r border-white/10 flex flex-col text-neutral-200 shadow-2xl z-20 flex-shrink-0 font-sans transition-all duration-300 ease-in-out`}>
      {/* Header */}
      <div className={`p-4 border-b border-white/10 bg-[#16181b] flex items-center ${isCollapsed ? 'justify-center flex-col gap-4' : 'justify-between'}`}>
        {!isCollapsed && (
          <div>
            <h1 className="text-xl font-bold flex items-center gap-2.5 tracking-tight text-white">
              <div className="p-1.5 bg-blue-500/10 rounded-lg">
                <Layers className="w-5 h-5 text-blue-500" />
              </div>
              GeoViewer
            </h1>
            <p className="text-[10px] uppercase tracking-wider text-neutral-500 mt-2 font-semibold pl-11">AutoCAD Visualizer</p>
          </div>
        )}
        
        <button 
          onClick={() => setIsCollapsed(!isCollapsed)}
          className="p-1.5 hover:bg-white/10 rounded-lg text-neutral-400 hover:text-white transition-colors"
          title={isCollapsed ? "Expand Panel" : "Collapse Panel"}
        >
          {isCollapsed ? <PanelLeftOpen className="w-5 h-5" /> : <PanelLeftClose className="w-5 h-5" />}
        </button>
      </div>

      <div className="flex-1 overflow-y-auto custom-scrollbar overflow-x-hidden">
        {/* File Import Section */}
        <div className={`p-4 border-b border-white/5 ${isCollapsed ? 'flex flex-col gap-3 items-center' : ''}`}>
          {isCollapsed ? (
            <>
              <button
                onClick={() => pdfInputRef.current?.click()}
                className="p-2.5 flex items-center justify-center bg-blue-600 hover:bg-blue-500 text-white rounded-lg transition-all shadow-lg shadow-blue-900/20 font-medium text-sm group border border-blue-500/50 relative"
              >
                <FileText className="w-4 h-4" />
                <div className="absolute left-full top-1/2 -translate-y-1/2 ml-3 px-2 py-1 bg-black/90 text-white text-xs rounded opacity-0 group-hover:opacity-100 whitespace-nowrap z-50 pointer-events-none transition-opacity border border-white/10">
                  Upload PDFs
                </div>
              </button>
              <button
                onClick={() => dxfInputRef.current?.click()}
                className="p-2.5 flex items-center justify-center bg-blue-600 hover:bg-blue-500 text-white rounded-lg transition-all shadow-lg shadow-blue-900/20 font-medium text-sm group border border-blue-500/50 relative"
              >
                <Layers className="w-4 h-4" />
                <div className="absolute left-full top-1/2 -translate-y-1/2 ml-3 px-2 py-1 bg-black/90 text-white text-xs rounded opacity-0 group-hover:opacity-100 whitespace-nowrap z-50 pointer-events-none transition-opacity border border-white/10">
                  Upload DXFs
                </div>
              </button>
            </>
          ) : (
            <div className="space-y-4">
              <div className="rounded-xl border border-white/10 bg-white/[0.03]">
                <div className="flex items-center justify-between px-3 py-2 border-b border-white/10">
                  <div className="flex items-center gap-2 text-sm text-neutral-200">
                    <FileText className="w-4 h-4 text-blue-400" />
                    PDF Documents
                  </div>
                  <button
                    onClick={() => pdfInputRef.current?.click()}
                    className="px-3 py-1.5 bg-white/5 hover:bg-white/10 text-xs text-neutral-200 rounded-md border border-white/10 transition-colors"
                  >
                    Upload
                  </button>
                </div>
                <div className="max-h-48 overflow-y-auto">
                  {pdfs.length === 0 ? (
                    <div className="px-3 py-3 text-xs text-neutral-500">No PDFs uploaded</div>
                  ) : (
                    <div className="py-1">
                      {pdfs.map((pdf) => {
                        const label = (pdf.pdf_source || pdf.filename || pdf.pdf_id).toString();
                        const isSelected = pdf.pdf_id === selectedPdfId;
                        const statusValue = typeof pdf.status === 'string' ? pdf.status : null;
                        const isProcessing = statusValue === 'processing';
                        const status = statusValue ? statusValue.toUpperCase() : null;
                        const details = pdf.status_details;
                        const detailParts: string[] = [];
                        if (isProcessing && details?.step) {
                          detailParts.push(details.step.replace(/_/g, ' '));
                        }
                        if (isProcessing && details?.progress) {
                          detailParts.push(`${details.progress} pages`);
                        }
                        if (isProcessing && !details?.progress && typeof details?.pages === 'number') {
                          detailParts.push(`${details.pages} pages`);
                        }
                        const detailLabel = isProcessing && detailParts.length > 0 ? detailParts.join(' • ') : null;
                        return (
                          <div
                            key={pdf.pdf_id}
                            className={`flex items-start gap-2 px-3 py-2 text-sm transition-colors ${isSelected ? 'bg-blue-500/15 text-blue-200' : 'text-neutral-300 hover:bg-white/5'}`}
                            title={label}
                          >
                            <button
                              type="button"
                              onClick={() => onSelectPdfId(pdf.pdf_id)}
                              className="flex-1 text-left min-w-0"
                            >
                              <span className="truncate block">{label}</span>
                              {status && (
                                <span className="text-[10px] text-neutral-500">
                                  {status}{detailLabel ? ` · ${detailLabel}` : ''}
                                </span>
                              )}
                            </button>
                            {onDeletePdf && (
                              <button
                                type="button"
                                onClick={() => onDeletePdf(pdf.pdf_id)}
                                className="p-1 rounded hover:bg-white/10 text-neutral-400 hover:text-red-400"
                                title="Remove"
                              >
                                <Trash2 className="w-3.5 h-3.5" />
                              </button>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              </div>

              <div className="rounded-xl border border-white/10 bg-white/[0.03]">
                <div className="flex items-center justify-between px-3 py-2 border-b border-white/10">
                  <div className="flex items-center gap-2 text-sm text-neutral-200">
                    <Layers className="w-4 h-4 text-blue-400" />
                    CAD Files (DXF)
                  </div>
                  <button
                    onClick={() => dxfInputRef.current?.click()}
                    className="px-3 py-1.5 bg-white/5 hover:bg-white/10 text-xs text-neutral-200 rounded-md border border-white/10 transition-colors"
                  >
                    Upload
                  </button>
                </div>
                <div className="max-h-48 overflow-y-auto">
                  {dxfs.length === 0 ? (
                    <div className="px-3 py-3 text-xs text-neutral-500">No DXFs uploaded</div>
                  ) : (
                    <div className="py-1">
                      {dxfs.map((dxf) => {
                        const label = (dxf.source_filename || dxf.filename || dxf.dxf_id).toString();
                        const isSelected = dxf.dxf_id === selectedDxfId;
                        const statusValue = typeof dxf.status === 'string' ? dxf.status : null;
                        const isProcessing = statusValue === 'processing';
                        const status = statusValue ? statusValue.toUpperCase() : null;
                        
                        const details = dxf.status_details;
                        const detailParts: string[] = [];
                        if (isProcessing && details?.step) {
                          detailParts.push(details.step.replace(/_/g, ' '));
                        }
                        if (isProcessing && details?.progress) {
                          detailParts.push(details.progress);
                        }
                        const detailLabel = isProcessing && detailParts.length > 0 ? detailParts.join(' • ') : null;

                        return (
                          <div
                            key={dxf.dxf_id}
                            className={`flex items-start gap-2 px-3 py-2 text-sm transition-colors ${isSelected ? 'bg-blue-500/15 text-blue-200' : 'text-neutral-300 hover:bg-white/5'}`}
                            title={label}
                          >
                            <button
                              type="button"
                              onClick={() => onSelectDxfId(dxf.dxf_id)}
                              className="flex-1 text-left min-w-0"
                            >
                              <span className="truncate block">{label}</span>
                              {status && (
                                <span className="text-[10px] text-neutral-500">
                                  {status}{detailLabel ? ` · ${detailLabel}` : ''}
                                </span>
                              )}
                            </button>
                            {onDeleteDxf && (
                              <button
                                type="button"
                                onClick={() => onDeleteDxf(dxf.dxf_id)}
                                className="p-1 rounded hover:bg-white/10 text-neutral-400 hover:text-red-400"
                                title="Remove"
                              >
                                <Trash2 className="w-3.5 h-3.5" />
                              </button>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          <input
            type="file"
            ref={pdfInputRef}
            onChange={handlePdfChange}
            accept=".pdf"
            multiple
            className="hidden"
          />
          <input
            type="file"
            ref={dxfInputRef}
            onChange={handleDxfChange}
            accept=".dxf"
            multiple
            className="hidden"
          />
        </div>

        {/* Collapsed View Mode Controls */}
        {isCollapsed && fileName && onViewModeChange && (
          <div className="p-2 border-b border-white/5 flex flex-col gap-2 items-center animate-in fade-in duration-300">
             {/* Overlay */}
             <div className="group relative">
               <button
                 onClick={() => onViewModeChange('overlay')}
                 className={`p-2.5 rounded-lg transition-all ${
                   viewMode === 'overlay'
                     ? 'bg-blue-600 text-white shadow-lg shadow-blue-900/20'
                     : 'text-neutral-400 hover:bg-white/5 hover:text-white'
                 }`}
               >
                 <Layers className="w-5 h-5" />
               </button>
               <div className="absolute left-full top-1/2 -translate-y-1/2 ml-3 px-2 py-1 bg-black/90 text-white text-xs rounded opacity-0 group-hover:opacity-100 whitespace-nowrap z-50 pointer-events-none transition-opacity border border-white/10">
                 Overlay View
               </div>
             </div>

             {/* Gallery */}
             <div className="group relative">
               <button
                 onClick={() => onViewModeChange('gallery')}
                 className={`p-2.5 rounded-lg transition-all ${
                   viewMode === 'gallery'
                     ? 'bg-blue-600 text-white shadow-lg shadow-blue-900/20'
                     : 'text-neutral-400 hover:bg-white/5 hover:text-white'
                 }`}
               >
                 <Grid className="w-5 h-5" />
               </button>
               <div className="absolute left-full top-1/2 -translate-y-1/2 ml-3 px-2 py-1 bg-black/90 text-white text-xs rounded opacity-0 group-hover:opacity-100 whitespace-nowrap z-50 pointer-events-none transition-opacity border border-white/10">
                 Gallery View
               </div>
             </div>

             {/* Graph */}
             <div className="group relative">
               <button
                 onClick={() => onViewModeChange('graph')}
                 className={`p-2.5 rounded-lg transition-all ${
                   viewMode === 'graph'
                     ? 'bg-blue-600 text-white shadow-lg shadow-blue-900/20'
                     : 'text-neutral-400 hover:bg-white/5 hover:text-white'
                 }`}
               >
                 <Network className="w-5 h-5" />
               </button>
               <div className="absolute left-full top-1/2 -translate-y-1/2 ml-3 px-2 py-1 bg-black/90 text-white text-xs rounded opacity-0 group-hover:opacity-100 whitespace-nowrap z-50 pointer-events-none transition-opacity border border-white/10">
                 Graph View
               </div>
             </div>

             {/* Raster */}
             <div className="group relative">
               <button
                 onClick={() => onViewModeChange('raster')}
                 className={`p-2.5 rounded-lg transition-all ${
                   viewMode === 'raster'
                     ? 'bg-blue-600 text-white shadow-lg shadow-blue-900/20'
                     : 'text-neutral-400 hover:bg-white/5 hover:text-white'
                 }`}
               >
                 <ImageIcon className="w-5 h-5" />
               </button>
               <div className="absolute left-full top-1/2 -translate-y-1/2 ml-3 px-2 py-1 bg-black/90 text-white text-xs rounded opacity-0 group-hover:opacity-100 whitespace-nowrap z-50 pointer-events-none transition-opacity border border-white/10">
                 Raster View
               </div>
             </div>
          </div>
        )}

        {fileName && (
          <Section title="Project Overview" icon={Box} isCollapsed={isCollapsed}>
            <div className="space-y-4">
              <div className="bg-white/5 rounded-lg p-3 border border-white/5 flex items-center gap-3">
                 <div className="relative">
                    <div className="w-2.5 h-2.5 rounded-full bg-green-500 animate-pulse" />
                    <div className="absolute inset-0 w-2.5 h-2.5 rounded-full bg-green-500 blur-sm opacity-50" />
                 </div>
                 <span className="text-sm font-medium truncate flex-1 text-neutral-200" title={fileName}>{fileName}</span>
              </div>
              
              {onViewModeChange && (
                 <div className="flex gap-1 p-1 bg-black/20 rounded-lg border border-white/5">
                   <button
                     onClick={() => onViewModeChange('overlay')}
                     className={`flex-1 flex items-center justify-center gap-2 py-1.5 rounded-md text-xs font-medium transition-all ${
                       viewMode === 'overlay' 
                         ? 'bg-blue-600 text-white shadow-lg shadow-blue-900/20' 
                         : 'text-neutral-400 hover:text-neutral-200 hover:bg-white/5'
                     }`}
                     title="Overlay View"
                   >
                     <Layers className="w-3.5 h-3.5" />
                     {!isCollapsed && "Overlay"}
                   </button>
                   <button
                     onClick={() => onViewModeChange('gallery')}
                     className={`flex-1 flex items-center justify-center gap-2 py-1.5 rounded-md text-xs font-medium transition-all ${
                       viewMode === 'gallery' 
                         ? 'bg-blue-600 text-white shadow-lg shadow-blue-900/20' 
                         : 'text-neutral-400 hover:text-neutral-200 hover:bg-white/5'
                     }`}
                     title="Gallery View"
                   >
                     <Grid className="w-3.5 h-3.5" />
                     {!isCollapsed && "Gallery"}
                   </button>
                    <button
                      onClick={() => onViewModeChange('graph')}
                      className={`flex-1 flex items-center justify-center gap-2 py-1.5 rounded-md text-xs font-medium transition-all ${
                        viewMode === 'graph' 
                          ? 'bg-blue-600 text-white shadow-lg shadow-blue-900/20' 
                          : 'text-neutral-400 hover:text-neutral-200 hover:bg-white/5'
                      }`}
                      title="Graph View"
                    >
                      <Network className="w-3.5 h-3.5" />
                      {!isCollapsed && "Graph"}
                    </button>
                    <button
                      onClick={() => onViewModeChange('raster')}
                      className={`flex-1 flex items-center justify-center gap-2 py-1.5 rounded-md text-xs font-medium transition-all ${
                        viewMode === 'raster' 
                          ? 'bg-blue-600 text-white shadow-lg shadow-blue-900/20' 
                          : 'text-neutral-400 hover:text-neutral-200 hover:bg-white/5'
                      }`}
                      title="Raster View"
                    >
                      <ImageIcon className="w-3.5 h-3.5" />
                      {!isCollapsed && "Raster"}
                    </button>
                 </div>
               )}

              {meta && (
                <div className="grid grid-cols-2 gap-3">
                  <div className="bg-white/5 p-3 rounded-lg border border-white/5 text-center group hover:border-white/10 transition-colors">
                    <div className="text-[10px] uppercase text-neutral-500 font-bold mb-1 group-hover:text-neutral-400 transition-colors">Layouts</div>
                    <div className="text-xl font-bold text-white">{meta.layouts}</div>
                  </div>
                  <div className="bg-white/5 p-3 rounded-lg border border-white/5 text-center group hover:border-white/10 transition-colors">
                    <div className="text-[10px] uppercase text-neutral-500 font-bold mb-1 group-hover:text-neutral-400 transition-colors">Chunks</div>
                    <div className="text-xl font-bold text-white">{meta.chunks}</div>
                  </div>
                </div>
              )}

              {/* Mask Toggle Control */}
              {viewMode === 'raster' && onToggleMasks && (
                <div className="mt-3 space-y-2">
                  <button
                    onClick={() => onToggleMasks(!showMasks)}
                    className={`w-full flex items-center justify-between p-3 rounded-lg border transition-all ${
                      showMasks 
                        ? 'bg-blue-500/10 border-blue-500/50 text-blue-400' 
                        : 'bg-white/5 border-white/10 text-neutral-400 hover:bg-white/10'
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      {showMasks ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
                      <span className="text-sm font-medium">Show Masks</span>
                    </div>
                    <div className={`w-8 h-4 rounded-full relative transition-colors ${showMasks ? 'bg-blue-500' : 'bg-neutral-600'}`}>
                      <div className={`absolute top-0.5 w-3 h-3 bg-white rounded-full transition-all ${showMasks ? 'left-4.5' : 'left-0.5'}`} style={{ left: showMasks ? '18px' : '2px' }} />
                    </div>
                  </button>
                </div>
              )}
            </div>
          </Section>
        )}

        {selectedFeature ? (
          <>
            <Section title="Identity" icon={Hash} isCollapsed={isCollapsed}>
              <div className="bg-white/5 rounded-lg px-4 py-2 border border-white/5">
                <DataRow
                  label="Layout ID"
                  value={typeof layoutId === 'number' || typeof layoutId === 'string'
                    ? layoutId
                    : '—'}
                  mono
                />
                <DataRow
                  label="Chunk ID"
                  value={typeof chunkId === 'number' || typeof chunkId === 'string'
                    ? chunkId
                    : '—'}
                  mono
                />
                <DataRow
                  label="Node ID"
                  value={typeof nodeId === 'number' || typeof nodeId === 'string'
                    ? nodeId
                    : '—'}
                  mono
                />
              </div>
            </Section>

            {Array.isArray(graphFeatures) && (
              <Section title="Graph Features" icon={Activity} isCollapsed={isCollapsed}>
                 <div className="bg-white/5 rounded-lg p-3 border border-white/5 space-y-2">
                    {(graphFeatures as unknown[]).map((f, i) => (
                      <div key={i} className="group relative">
                        <div className="flex justify-between items-end mb-1">
                          <span className="text-[10px] text-neutral-500 font-mono font-bold">f{i}</span>
                          <span className="text-xs font-mono text-neutral-300">{typeof f === 'number' ? f.toFixed(4) : String(f)}</span>
                        </div>
                        <div className="h-1.5 w-full bg-black/40 rounded-full overflow-hidden border border-white/5">
                          <div 
                            className="h-full bg-blue-500 rounded-full transition-all duration-500 ease-out"
                            style={{ width: `${Math.min(Math.max((typeof f === 'number' ? f : 0) * 100, 0), 100)}%` }} 
                          />
                        </div>
                      </div>
                    ))}
                 </div>
              </Section>
            )}

            {Array.isArray(neighbors) && (
              <Section title={`Neighbors (${neighbors.length})`} icon={Share2} isCollapsed={isCollapsed}>
                <div className="flex flex-wrap gap-2">
                  {(neighbors as unknown[]).map((id) => (
                    <Tag key={String(id)}>{String(id)}</Tag>
                  ))}
                </div>
              </Section>
            )}

            {selectedFeature.geometry && (
              <Section title="Geometry" icon={Map} defaultOpen={false} isCollapsed={isCollapsed}>
                <div className="space-y-3">
                   <div className="flex justify-between text-xs text-neutral-400 px-1">
                      <span>Type</span>
                      <span className="text-white font-medium">{selectedFeature.geometry.type ?? '—'}</span>
                   </div>
                   {selectedFeature.geometry.type === 'Polygon' && Array.isArray(selectedFeature.geometry.coordinates) && Array.isArray(selectedFeature.geometry.coordinates[0]) && (
                     <div className="flex justify-between text-xs text-neutral-400 px-1">
                        <span>Vertices</span>
                        <span className="text-white font-medium">{(selectedFeature.geometry.coordinates[0] as unknown[]).length}</span>
                     </div>
                   )}
                   <div className="relative group">
                     <div className="absolute right-2 top-2 opacity-0 group-hover:opacity-100 transition-opacity">
                        {geometryCoordinates && <CopyButton text={geometryCoordinatesText} />}
                     </div>
                     <div className="text-[10px] bg-black/30 p-3 rounded-lg border border-white/10 font-mono text-neutral-400 h-60 overflow-y-auto w-full overflow-x-hidden whitespace-pre-wrap break-all overscroll-contain shadow-inner">
                        {geometryCoordinatesText}
                     </div>
                   </div>
                </div>
              </Section>
            )}
          </>
        ) : (
          !isCollapsed && (
            <div className="p-8 mt-8 text-center text-neutral-600 flex flex-col items-center gap-4">
               <div className="w-16 h-16 rounded-full bg-white/5 flex items-center justify-center">
                  <Info className="w-8 h-8 opacity-20" />
               </div>
               <div className="space-y-1">
                 <p className="text-sm font-medium text-neutral-400">No Feature Selected</p>
                 <p className="text-xs text-neutral-600">Select a chunk in the viewer to inspect properties.</p>
               </div>
            </div>
          )
        )}
      </div>
      
      {/* Footer Controls Info */}
      {!isCollapsed && (
        <div className="p-4 border-t border-white/10 bg-[#16181b] text-[11px] text-neutral-500 font-medium animate-in fade-in duration-300">
          <div className="flex items-center justify-between mb-2">
            <span className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-blue-500/50" />
              Pan
            </span>
            <span className="text-neutral-600">Left Click + Drag</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-blue-500/50" />
              Zoom
            </span>
            <span className="text-neutral-600">Scroll</span>
          </div>
        </div>
      )}
    </div>
  );
}
