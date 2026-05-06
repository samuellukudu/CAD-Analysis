import { Home, Settings, Layers, Grid, Network, Image as ImageIcon, Menu, ChevronDown, FileText, Play, MessageCircle } from 'lucide-react';
import MultiSelect from './ui/MultiSelect';

type RibbonTab = 'home' | 'view' | 'analysis' | 'assign';

interface TopNavProps {
  viewMode: 'overlay' | 'gallery' | 'graph' | 'raster';
  onViewModeChange: (mode: 'overlay' | 'gallery' | 'graph' | 'raster') => void;
  ribbonTab: RibbonTab;
  onRibbonTabChange: (tab: RibbonTab) => void;
  agentQuery: string;
  onAgentQueryChange: (query: string) => void;
  onRunAgentAnalysis: () => void;
  fileName?: string | null;
  layoutOptions: { label: string, value: number }[];
  chunkOptions: { label: string, value: number }[];
  selectedLayouts: number[];
  onLayoutsChange: (ids: number[]) => void;
  selectedChunks: number[];
  onChunksChange: (ids: number[]) => void;
  hasPdf?: boolean;
  showPdfPanel?: boolean;
  onTogglePdfPanel?: (show: boolean) => void;
  onToggleAgentPanel?: (show: boolean) => void;
  showAgentPanel?: boolean;
  onToggleChatPanel?: (show: boolean) => void;
  showChatPanel?: boolean;
}

export default function TopNav({
  viewMode,
  onViewModeChange,
  ribbonTab,
  onRibbonTabChange,
  agentQuery,
  onAgentQueryChange,
  onRunAgentAnalysis,
  fileName,
  layoutOptions,
  chunkOptions,
  selectedLayouts,
  onLayoutsChange,
  selectedChunks,
  onChunksChange,
  hasPdf,
  showPdfPanel,
  onTogglePdfPanel,
  onToggleAgentPanel,
  showAgentPanel,
  onToggleChatPanel,
  showChatPanel
}: TopNavProps) {
  const tabs: RibbonTab[] = ['home', 'view', 'analysis', 'assign'];

  return (
    <div className="flex flex-col w-full bg-[#16181b] border-b border-white/10 z-30">
      {/* Top Bar (Logo & Window Controls area) */}
      <div className="h-8 flex items-center justify-between px-2 bg-[#0d0e10] border-b border-white/5 select-none">
        <div className="flex items-center gap-3">
          <div className="text-red-500 font-bold text-lg tracking-tighter">A</div>
          <div className="h-4 w-[1px] bg-white/10 mx-1" />
          <div className="flex items-center gap-2 text-xs text-neutral-400 hover:text-white cursor-pointer transition-colors">
            <Menu className="w-3.5 h-3.5" />
            <ChevronDown className="w-3 h-3" />
          </div>
          {fileName && (
            <div className="flex items-center gap-2 ml-4">
              <span className="text-xs text-neutral-500">Running object snaps</span>
              <span className="text-xs text-neutral-600">|</span>
              <span className="text-xs text-neutral-300">{fileName}</span>
            </div>
          )}
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => onToggleChatPanel?.(!showChatPanel)}
            className={`p-1.5 rounded hover:bg-white/10 transition-colors ${showChatPanel ? 'text-blue-400 bg-blue-500/10' : 'text-neutral-400'}`}
            title="Open Chat"
          >
            <MessageCircle className="w-3.5 h-3.5" />
          </button>
          <div className="text-[10px] text-neutral-600 font-mono">
            AutoCAD Web Style Visualizer
          </div>
        </div>
      </div>

      {/* Ribbon Tabs */}
      <div className="flex items-center px-1 border-b border-white/5 bg-[#16181b]">
        {tabs.map((tab) => (
          <button
            key={tab}
            onClick={() => onRibbonTabChange(tab)}
            className={`px-4 py-1.5 text-xs font-medium transition-colors border-t-2 capitalize ${ribbonTab === tab
              ? 'border-blue-500 text-white bg-white/5'
              : 'border-transparent text-neutral-400 hover:text-neutral-200 hover:bg-white/[0.02]'
              }`}
          >
            {tab}
          </button>
        ))}
      </div>

      {/* Ribbon Content (Toolbar) */}
      <div className="h-20 bg-[#1a1d21] border-b border-white/10 px-4 py-2 flex items-center gap-6 overflow-x-auto custom-scrollbar">
        {ribbonTab === 'view' && (
          <>
            {/* View Modes Group */}
            <div className="flex flex-col h-full justify-between group">
              <div className="flex gap-1">
                <button
                  onClick={() => onViewModeChange('overlay')}
                  className={`flex flex-col items-center gap-1 p-2 rounded hover:bg-white/10 transition-all ${viewMode === 'overlay' ? 'bg-blue-500/20 text-blue-400' : 'text-neutral-400'}`}
                  title="Overlay View"
                >
                  <Layers className="w-6 h-6" />
                  <span className="text-[10px] font-medium">Overlay</span>
                </button>
                <button
                  onClick={() => onViewModeChange('gallery')}
                  className={`flex flex-col items-center gap-1 p-2 rounded hover:bg-white/10 transition-all ${viewMode === 'gallery' ? 'bg-blue-500/20 text-blue-400' : 'text-neutral-400'}`}
                  title="Gallery View"
                >
                  <Grid className="w-6 h-6" />
                  <span className="text-[10px] font-medium">Gallery</span>
                </button>
              </div>
              <div className="text-[10px] text-neutral-600 text-center font-medium uppercase tracking-wider border-t border-white/5 pt-0.5 mt-auto">Modes</div>
            </div>

            <div className="w-[1px] h-full bg-white/10" />

            {/* Analysis Views Group */}
            <div className="flex flex-col h-full justify-between group">
              <div className="flex gap-1">
                <button
                  onClick={() => onViewModeChange('graph')}
                  className={`flex flex-col items-center gap-1 p-2 rounded hover:bg-white/10 transition-all ${viewMode === 'graph' ? 'bg-blue-500/20 text-blue-400' : 'text-neutral-400'}`}
                  title="Graph View"
                >
                  <Network className="w-6 h-6" />
                  <span className="text-[10px] font-medium">Graph</span>
                </button>
                <button
                  onClick={() => onViewModeChange('raster')}
                  className={`flex flex-col items-center gap-1 p-2 rounded hover:bg-white/10 transition-all ${viewMode === 'raster' ? 'bg-blue-500/20 text-blue-400' : 'text-neutral-400'}`}
                  title="Raster View"
                >
                  <ImageIcon className="w-6 h-6" />
                  <span className="text-[10px] font-medium">Raster</span>
                </button>
              </div>
              <div className="text-[10px] text-neutral-600 text-center font-medium uppercase tracking-wider border-t border-white/5 pt-0.5 mt-auto">Analysis</div>
            </div>

            <div className="w-[1px] h-full bg-white/10" />

            {/* Tools Group */}
            {hasPdf && (
              <>
                <div className="flex flex-col h-full justify-between group">
                  <div className="flex gap-1">
                    <button
                      onClick={() => onTogglePdfPanel?.(!showPdfPanel)}
                      className={`flex flex-col items-center gap-1 p-2 rounded hover:bg-white/10 transition-all ${showPdfPanel ? 'bg-blue-500/20 text-blue-400' : 'text-neutral-400'}`}
                      title={showPdfPanel ? "Hide PDF Panel" : "Show PDF Panel"}
                    >
                      <FileText className="w-6 h-6" />
                      <span className="text-[10px] font-medium">PDF Panel</span>
                    </button>
                  </div>
                  <div className="text-[10px] text-neutral-600 text-center font-medium uppercase tracking-wider border-t border-white/5 pt-0.5 mt-auto">Tools</div>
                </div>
                <div className="w-[1px] h-full bg-white/10" />
              </>
            )}

            {/* Filters Group */}
            <div className="flex flex-col h-full justify-between group">
              <div className="flex gap-1">
                <MultiSelect
                  options={layoutOptions}
                  value={selectedLayouts}
                  onChange={onLayoutsChange}
                  placeholder="All Layouts"
                  label="Layouts"
                  icon={Layers}
                  className="w-auto"
                />
                <MultiSelect
                  options={chunkOptions}
                  value={selectedChunks}
                  onChange={onChunksChange}
                  placeholder="All Chunks"
                  label="Chunks"
                  icon={Grid}
                  className="w-auto"
                />
              </div>
              <div className="text-[10px] text-neutral-600 text-center font-medium uppercase tracking-wider border-t border-white/5 pt-0.5 mt-auto">Filters</div>
            </div>
          </>
        )}

        {ribbonTab === 'home' && (
          <div className="flex items-center justify-center w-full text-neutral-500 text-sm italic">
            <Home className="w-4 h-4 mr-2" />
            Common tools will appear here
          </div>
        )}

        {ribbonTab === 'analysis' && (
          <div className="flex flex-col h-full justify-between group">
            <div className="flex gap-1">
              <button
                onClick={() => onToggleAgentPanel?.(!showAgentPanel)}
                className={`flex flex-col items-center gap-1 p-2 rounded hover:bg-white/10 transition-all ${showAgentPanel ? 'bg-purple-500/20 text-purple-400' : 'text-neutral-400'}`}
                title="Open Agent Results"
              >
                <Settings className="w-6 h-6" />
                <span className="text-[10px] font-medium">Agent</span>
              </button>
            </div>
            <div className="text-[10px] text-neutral-600 text-center font-medium uppercase tracking-wider border-t border-white/5 pt-0.5 mt-auto">Results</div>
          </div>
        )}

        {ribbonTab === 'assign' && (
          <div className="flex items-center justify-between w-full gap-4">
            <div className="flex flex-col w-1/2 min-w-[240px] max-w-[520px]">
              <textarea
                value={agentQuery}
                onChange={(e) => onAgentQueryChange(e.target.value)}
                rows={2}
                className="w-full min-h-[48px] max-h-[72px] px-3 py-2 rounded-lg bg-white border border-neutral-300 text-sm text-black placeholder:text-neutral-500 focus:outline-none focus:bg-white focus:border-purple-500 transition-all resize-none"
                placeholder="Enter your query..."
              />
            </div>
            <button
              onClick={onRunAgentAnalysis}
              className="h-10 px-4 rounded-md bg-purple-600 hover:bg-purple-500 text-white text-sm font-medium shadow-lg shadow-purple-900/20 flex items-center gap-2 transition-all"
              title="Run Agent Analysis"
            >
              <Play className="w-4 h-4 fill-white" />
              Run
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
