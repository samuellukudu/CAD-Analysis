import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { X, Search, FileText, Loader2, ChevronRight, Image as ImageIcon } from 'lucide-react';
import { API_ENDPOINTS } from '../config/api';

interface PdfListItem {
  pdf_id: string;
  filename: string;
  pdf_source?: string | null;
}

interface PdfPanelProps {
  pdfId: string | null;
  pdfs: PdfListItem[];
  onSelectPdfId: (pdfId: string) => void;
  onClose: () => void;
}

interface SearchResult {
  page_index: number;
  page_number: number;
  pdf_source: string;
  pdf_id?: string;
  similarity_score: number;
  image_base64?: string;
  width?: number;
  height?: number;
}

export default function PdfPanel({ pdfId, pdfs, onSelectPdfId, onClose }: PdfPanelProps) {
  const MIN_QUERY_LEN = 2;
  const [query, setQuery] = useState('');
  const [isSearching, setIsSearching] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [pdfUrl, setPdfUrl] = useState<string | null>(null);
  const [isLoadingPdf, setIsLoadingPdf] = useState(false);
  const [pendingJumpPageNumber, setPendingJumpPageNumber] = useState<number | null>(null);
  const [searchScope, setSearchScope] = useState<'all' | 'selected'>('all');
  const [autoSearch, setAutoSearch] = useState(true);
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const searchAbortRef = useRef<AbortController | null>(null);

  const isSearchMode = isSearching || hasSearched;

  const selectedPdfLabel = useMemo(() => {
    if (!pdfId) return null;
    const item = pdfs.find((p) => p.pdf_id === pdfId);
    return item ? (item.pdf_source || item.filename || item.pdf_id) : pdfId;
  }, [pdfId, pdfs]);

  useEffect(() => {
    if (!pdfId) {
      setIsLoadingPdf(false);
      setPdfUrl(null);
      return;
    }

    setIsLoadingPdf(true);
    setPdfUrl(API_ENDPOINTS.GET_PDF_CONTENT(pdfId));
    const timer = window.setTimeout(() => setIsLoadingPdf(false), 150);

    return () => window.clearTimeout(timer);
  }, [pdfId]);

  useEffect(() => {
    if (!pendingJumpPageNumber || !pdfUrl) return;
    if (iframeRef.current) {
      iframeRef.current.src = `${pdfUrl}#page=${pendingJumpPageNumber}&view=Fit`;
    }
    setPendingJumpPageNumber(null);
  }, [pendingJumpPageNumber, pdfUrl]);

  const performSearch = useCallback(async (nextQuery: string) => {
    const trimmed = nextQuery.trim();
    if (!trimmed) return;
    if (trimmed.length < MIN_QUERY_LEN) {
      setIsSearching(false);
      setHasSearched(true);
      setSearchError(`Type at least ${MIN_QUERY_LEN} characters to search.`);
      setResults([]);
      return;
    }

    if (searchScope === 'selected' && !pdfId) {
      setIsSearching(false);
      setHasSearched(true);
      setSearchError('Select a PDF first to search within it.');
      setResults([]);
      return;
    }

    searchAbortRef.current?.abort();
    const abortController = new AbortController();
    searchAbortRef.current = abortController;

    setIsSearching(true);
    setHasSearched(true);
    setSearchError(null);
    if (results.length === 0) setResults([]);
    try {
      const res = await fetch(`${API_ENDPOINTS.SEARCH_PDFS}?query=${encodeURIComponent(trimmed)}`, {
        signal: abortController.signal
      });
      if (!res.ok) throw new Error('Search failed');
      const data = await res.json();
      const raw = (data.results || []) as SearchResult[];
      const filtered = searchScope === 'selected' && pdfId
        ? raw.filter((r) => r.pdf_id === pdfId)
        : raw;
      setResults(filtered);
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') return;
      console.error('Search error:', error);
      setSearchError('Search failed. Please try again.');
    } finally {
      setIsSearching(false);
    }
  }, [MIN_QUERY_LEN, pdfId, results.length, searchScope]);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    await performSearch(query);
  };

  const jumpToPage = (pageNumber: number) => {
    if (iframeRef.current && pdfUrl) {
      // Reload iframe with page hash
      // Note: This often requires the iframe src to be reset or postMessage if using a specialized viewer.
      // With native PDF viewer in iframe, changing src usually works.
      iframeRef.current.src = `${pdfUrl}#page=${pageNumber}&view=Fit`;
    }
  };

  const clearSearch = () => {
    searchAbortRef.current?.abort();
    setIsSearching(false);
    setHasSearched(false);
    setSearchError(null);
    setResults([]);
    setQuery('');
  };

  useEffect(() => {
    if (!autoSearch) return;
    const trimmed = query.trim();
    if (!trimmed) return;
    if (trimmed.length < MIN_QUERY_LEN) return;

    const timer = window.setTimeout(() => {
      performSearch(trimmed);
    }, 300);

    return () => window.clearTimeout(timer);
  }, [autoSearch, query, performSearch]);

  return (
    <div className="w-[500px] h-full bg-[#1a1d21] border-l border-white/10 flex flex-col shadow-2xl z-20 transition-all duration-300">
      {/* Header */}
      <div className="p-4 border-b border-white/10 bg-[#16181b] flex items-center justify-between">
        <div className="min-w-0">
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <FileText className="w-5 h-5 text-blue-500" />
            PDF Viewer
          </h2>
          <div className="text-xs text-neutral-500 truncate">
            {pdfId ? (selectedPdfLabel || pdfId) : (pdfs.length > 0 ? 'Select a PDF to preview' : 'No PDFs available')}
          </div>
        </div>
        <button 
          onClick={onClose}
          className="p-1.5 hover:bg-white/10 rounded-lg text-neutral-400 hover:text-white transition-colors"
        >
          <X className="w-5 h-5" />
        </button>
      </div>

      {/* Controls */}
      <div className="p-4 border-b border-white/5 space-y-3">
        <select
          value={pdfId ?? ''}
          onChange={(e) => {
            const next = e.target.value;
            if (!next) return;
            clearSearch();
            onSelectPdfId(next);
          }}
          className="w-full bg-black/20 border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500/50 transition-colors"
        >
          <option value="" disabled>
            {pdfs.length > 0 ? 'Choose a PDF…' : 'No PDFs found'}
          </option>
          {pdfs.map((p) => (
            <option key={p.pdf_id} value={p.pdf_id}>
              {(p.pdf_source || p.filename || p.pdf_id).toString()}
            </option>
          ))}
        </select>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setSearchScope('all')}
            className={`px-2.5 py-1 rounded-md text-xs border transition-colors ${searchScope === 'all' ? 'bg-blue-500/15 text-blue-300 border-blue-500/30' : 'bg-black/10 text-neutral-400 border-white/10 hover:border-white/20'}`}
          >
            All PDFs
          </button>
          <button
            type="button"
            onClick={() => setSearchScope('selected')}
            disabled={!pdfId}
            className={`px-2.5 py-1 rounded-md text-xs border transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${searchScope === 'selected' ? 'bg-blue-500/15 text-blue-300 border-blue-500/30' : 'bg-black/10 text-neutral-400 border-white/10 hover:border-white/20'}`}
          >
            This PDF
          </button>
          <div className="flex-1" />
          <button
            type="button"
            onClick={() => setAutoSearch((v) => !v)}
            className={`px-2.5 py-1 rounded-md text-xs border transition-colors ${autoSearch ? 'bg-white/5 text-neutral-300 border-white/15' : 'bg-black/10 text-neutral-400 border-white/10 hover:border-white/20'}`}
            title={autoSearch ? 'Auto search on typing' : 'Manual search (press Enter)'}
          >
            {autoSearch ? 'Auto' : 'Manual'}
          </button>
          <button
            type="button"
            onClick={() => clearSearch()}
            className="px-2.5 py-1 rounded-md text-xs border bg-black/10 text-neutral-400 border-white/10 hover:border-white/20 transition-colors"
            disabled={!query && !hasSearched}
          >
            Reset
          </button>
        </div>

        <form onSubmit={handleSearch} aria-busy={isSearching}>
          <div className="w-full flex items-stretch bg-white/[0.06] border border-white/10 rounded-lg overflow-hidden focus-within:border-blue-500/50 transition-colors text-neutral-100">
            <div className="flex items-center px-3 text-neutral-400">
              <Search className="w-3.5 h-3.5" />
            </div>
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Escape') clearSearch();
              }}
              placeholder="Search building codes & PDFs…"
              className="flex-1 min-w-0 bg-transparent px-0 py-2 text-sm !text-neutral-100 placeholder:text-neutral-500 caret-neutral-100 focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed selection:bg-blue-500/30 ![color:rgb(245_245_245)] ![-webkit-text-fill-color:rgb(245_245_245)] ![caret-color:rgb(245_245_245)]"
              disabled={pdfs.length === 0}
            />
            {!!query.trim() && !isSearching && (
              <button
                type="button"
                onClick={() => {
                  setQuery('');
                  setHasSearched(false);
                  setSearchError(null);
                  setResults([]);
                }}
                className="px-2 text-neutral-300 hover:text-white hover:bg-white/10 transition-colors"
                title="Clear"
              >
                <X className="w-4 h-4" />
              </button>
            )}
            <button
              type="submit"
              disabled={pdfs.length === 0 || isSearching}
              className="px-3 text-neutral-300 hover:text-white hover:bg-white/10 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              title="Search (Enter)"
            >
              {isSearching ? <Loader2 className="w-4 h-4 text-blue-500 animate-spin" /> : <Search className="w-4 h-4" />}
            </button>
          </div>
        </form>
        <div className="text-[11px] text-neutral-500 flex items-center justify-between">
          <span className="truncate">
            {searchScope === 'selected' && selectedPdfLabel ? `Searching within: ${selectedPdfLabel}` : 'Semantic search across indexed PDFs'}
          </span>
          <span className="ml-2 flex-shrink-0">Enter to search • Esc to clear</span>
        </div>
        {isSearching && (
          <div className="mt-2 h-0.5 w-full overflow-hidden rounded bg-white/10">
            <div className="h-full w-1/3 animate-pulse rounded bg-blue-500/70" />
          </div>
        )}
      </div>

      {/* Content Area */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Results List (Collapsible or Split) */}
        {isSearchMode && (
          <div
            className="flex-1 overflow-y-auto bg-[#111]"
          >
            <div className="p-2 space-y-2">
              <div className="px-2 py-1 text-xs font-semibold text-neutral-500 uppercase tracking-wider flex items-center justify-between">
                <span>Search Results</span>
                {isSearching ? (
                  <span className="inline-flex items-center gap-2 text-blue-400">
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    Searching…
                  </span>
                ) : (
                  <span>{results.length}</span>
                )}
              </div>

              {searchError && (
                <div className="px-2 py-6 text-sm text-red-400" role="alert">
                  {searchError}
                </div>
              )}

              {!searchError && isSearching && (
                <div className="space-y-2">
                  {[0, 1, 2].map((k) => (
                    <div key={k} className="p-3 rounded-lg border border-white/5 bg-white/[0.03] animate-pulse">
                      <div className="flex items-start gap-3">
                        <div className="w-16 h-16 rounded bg-black/30 border border-white/10 flex-shrink-0" />
                        <div className="flex-1 min-w-0 space-y-2">
                          <div className="h-4 w-28 rounded bg-white/10" />
                          <div className="h-3 w-3/4 rounded bg-white/10" />
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {!searchError && !isSearching && results.length === 0 && (
                <div className="px-2 py-6 text-sm text-neutral-500" aria-live="polite">
                  No results found.
                </div>
              )}

              {!searchError &&
                !isSearching &&
                results.map((result, idx) => (
                  <button
                    key={idx}
                    onClick={() => {
                      const targetPdfId = result.pdf_id;
                      if (targetPdfId && targetPdfId !== pdfId) {
                        onSelectPdfId(targetPdfId);
                        setPendingJumpPageNumber(result.page_number);
                      } else {
                        jumpToPage(result.page_number);
                      }
                      setHasSearched(false);
                    }}
                    className="w-full text-left p-3 rounded-lg hover:bg-white/5 border border-transparent hover:border-white/5 transition-all group"
                  >
                    <div className="flex items-start gap-3">
                      {result.image_base64 ? (
                        <div className="w-16 h-16 bg-black/40 rounded border border-white/10 overflow-hidden flex-shrink-0">
                          <img
                            src={`data:image/png;base64,${result.image_base64}`}
                            alt={`Page ${result.page_number}`}
                            className="w-full h-full object-cover"
                          />
                        </div>
                      ) : (
                        <div className="w-16 h-16 bg-white/5 rounded border border-white/10 flex items-center justify-center flex-shrink-0">
                          <ImageIcon className="w-6 h-6 text-neutral-600" />
                        </div>
                      )}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-sm font-medium text-neutral-200 truncate">
                            Page {result.page_number}
                          </span>
                          <span className="text-xs font-mono text-blue-400">
                            {Math.round(result.similarity_score * 100)}%
                          </span>
                        </div>
                        <p className="text-xs text-neutral-500 truncate">{result.pdf_source}</p>
                      </div>
                      <ChevronRight className="w-4 h-4 text-neutral-600 group-hover:text-neutral-300 mt-1" />
                    </div>
                  </button>
                ))}
            </div>
          </div>
        )}

        {/* PDF Viewer */}
        {!isSearchMode && (
          <div className="flex-1 bg-neutral-900 relative">
            {isLoadingPdf ? (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="flex flex-col items-center gap-3">
                  <Loader2 className="w-8 h-8 text-blue-500 animate-spin" />
                  <span className="text-sm text-neutral-400">Loading PDF...</span>
                </div>
              </div>
            ) : pdfUrl ? (
              <iframe
                ref={iframeRef}
                src={pdfUrl}
                className="w-full h-full border-0"
                title="PDF Preview"
              />
            ) : (
              <div className="absolute inset-0 flex items-center justify-center text-neutral-500">
                No PDF loaded
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
