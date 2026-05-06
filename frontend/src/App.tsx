import { useState, useMemo, useRef, useEffect, useCallback } from 'react';
import Viewer from './components/Viewer';
import ControlPanel from './components/ControlPanel';
import TopNav from './components/TopNav';
import PdfPanel from './components/PdfPanel';
import AgentPanel from './components/AgentPanel';
import ChatPanel from './components/ChatPanel';
import { calculateBounds, generateGalleryLayout, generateGraphData } from './utils/geoUtils';
import type { Bounds, GraphData, FeatureCollectionLike, FeatureLike } from './utils/geoUtils';
import { Loader2 } from 'lucide-react';
import { API_ENDPOINTS } from './config/api';
import type { Defect } from './services/agentService';

type PdfStatusDetails = {
  step?: string;
  progress?: string;
  pages?: number;
  index_status?: string;
  error?: string;
};

type PdfListItem = {
  pdf_id: string;
  filename: string;
  pdf_source?: string | null;
  status?: string | null;
  status_details?: PdfStatusDetails;
};

function App() {
  const uploadSeqRef = useRef(0);
  const [data, setData] = useState<FeatureCollectionLike | null>(null);
  const [bounds, setBounds] = useState<Bounds | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [meta, setMeta] = useState<{ chunks: number; layouts: number } | null>(null);
  const [selectedFeature, setSelectedFeature] = useState<FeatureLike | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [dxfId, setDxfId] = useState<string | null>(null);
  const [loadingMessage, setLoadingMessage] = useState<string>('Please wait while we analyze the file...');

  // PDF State
  const [pdfId, setPdfId] = useState<string | null>(null);
  const [showPdfPanel, setShowPdfPanel] = useState(false);
  const [availablePdfs, setAvailablePdfs] = useState<PdfListItem[]>([]);
  const availablePdfsRef = useRef<PdfListItem[]>([]);
  const [availableDxfs, setAvailableDxfs] = useState<Array<{ 
    dxf_id: string; 
    filename: string; 
    source_filename?: string | null; 
    status?: string | null;
    status_details?: { step?: string; progress?: string; error?: string };
  }>>([]);
  const availableDxfsRef = useRef<Array<{ 
    dxf_id: string; 
    filename: string; 
    source_filename?: string | null; 
    status?: string | null;
    status_details?: { step?: string; progress?: string; error?: string };
  }>>([]);

  // Agent State
  const [showAgentPanel, setShowAgentPanel] = useState(false);
  const [showChatPanel, setShowChatPanel] = useState(false);
  const [ribbonTab, setRibbonTab] = useState<'home' | 'view' | 'analysis' | 'assign'>('view');
  const [agentQuery, setAgentQuery] = useState<string>('what possible design and compliance issues are available in this drawing? ');
  const [agentDefects, setAgentDefects] = useState<Defect[]>([]);
  const [defectSeverityFilter, setDefectSeverityFilter] = useState<'all' | 'high' | 'medium' | 'low'>('all');
  const [agentRunId, setAgentRunId] = useState(0);

  // New State
  const [viewMode, setViewMode] = useState<'overlay' | 'gallery' | 'graph' | 'raster'>('overlay');
  const [galleryData, setGalleryData] = useState<FeatureCollectionLike | null>(null);
  const enrichPdfItems = useCallback(async (items: PdfListItem[]) => {
    const processing = items.filter((pdf) => pdf.status === 'processing');
    if (processing.length === 0) return items;
    const detailed = await Promise.all(
      processing.map(async (pdf) => {
        try {
          const statusRes = await fetch(API_ENDPOINTS.GET_PDF_STATUS(pdf.pdf_id));
          if (!statusRes.ok) return pdf;
          const statusData = await statusRes.json();
          const status = typeof statusData?.status === 'string' ? statusData.status : pdf.status;
          const details = statusData?.details && typeof statusData.details === 'object'
            ? (statusData.details as PdfStatusDetails)
            : undefined;
          return { ...pdf, status, status_details: details };
        } catch {
          return pdf;
        }
      })
    );
    const map = new Map(detailed.map((pdf) => [pdf.pdf_id, pdf]));
    return items.map((pdf) => map.get(pdf.pdf_id) ?? pdf);
  }, []);
  useEffect(() => {
    availablePdfsRef.current = availablePdfs;
  }, [availablePdfs]);

  useEffect(() => {
    availableDxfsRef.current = availableDxfs;
  }, [availableDxfs]);

  const refreshPdfList = useCallback(async (
    preferredPdfId?: string | null,
    fallbackPdfId?: string | null,
    options?: { select?: boolean }
  ) => {
    try {
      const res = await fetch(API_ENDPOINTS.LIST_PDFS);
      if (!res.ok) return [];
      const data = await res.json();
      const pdfs = (data?.pdfs || []) as PdfListItem[];
      const pending = availablePdfsRef.current.filter(
        (item) => item.status === 'processing' && !pdfs.some((pdf) => pdf.pdf_id === item.pdf_id)
      );
      const enriched = await enrichPdfItems([...pending, ...pdfs]);
      setAvailablePdfs(enriched);

      if (options && options.select === false) {
        return enriched;
      }

      const stored = (() => {
        try {
          return localStorage.getItem('selectedPdfId');
        } catch {
          return null;
        }
      })();

      const candidate = preferredPdfId || stored || fallbackPdfId;
      if (candidate && enriched.some((p) => p.pdf_id === candidate)) {
        setPdfId(candidate);
        return enriched;
      }
      if (enriched.length > 0) {
        setPdfId(enriched[0].pdf_id);
      }
      return enriched;
    } catch {
      return [];
    }
  }, [enrichPdfItems]);

  useEffect(() => {
    refreshPdfList(null, null);
  }, [refreshPdfList]);

  useEffect(() => {
    if (availablePdfs.every((pdf) => pdf.status !== 'processing')) return;
    const timer = window.setInterval(() => {
      refreshPdfList(null, null, { select: false });
    }, 2000);

    return () => window.clearInterval(timer);
  }, [availablePdfs, refreshPdfList]);

  useEffect(() => {
    if (!pdfId) return;
    try {
      localStorage.setItem('selectedPdfId', pdfId);
    } catch {
      return;
    }
  }, [pdfId]);
  useEffect(() => {
    if (!dxfId) return;
    try {
      localStorage.setItem('selectedDxfId', dxfId);
    } catch {
      return;
    }
  }, [dxfId]);
  const [galleryBounds, setGalleryBounds] = useState<Bounds | null>(null);
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [masksData, setMasksData] = useState<FeatureCollectionLike | null>(null);
  const [showMasks, setShowMasks] = useState(true);
  const [maskColorMode, setMaskColorMode] = useState<'single' | 'distinct'>('distinct');

  // Search/Filter State
  const [selectedLayouts, setSelectedLayouts] = useState<number[]>([]);
  const [selectedChunks, setSelectedChunks] = useState<number[]>([]);

  // Derived state for options
  const layoutOptions = useMemo(() => {
    if (!masksData) return [];
    const layouts = new Set<number>();
    masksData.features.forEach((f) => {
      const props = f.properties;
      const layoutIdRaw = props && typeof props === 'object' ? (props as Record<string, unknown>).layout_id : undefined;
      const layoutId = typeof layoutIdRaw === 'number' ? layoutIdRaw : Number(layoutIdRaw ?? NaN);
      if (Number.isFinite(layoutId)) layouts.add(layoutId);
    });
    return Array.from(layouts).sort((a, b) => a - b).map(id => ({ label: `Layout ${id}`, value: id }));
  }, [masksData]);

  const chunkOptions = useMemo(() => {
    if (!masksData) return [];
    const chunks = new Set<number>();

    // Filter features based on selected layouts first
    const relevantFeatures = selectedLayouts.length > 0
      ? masksData.features.filter((f) => {
        const props = f.properties;
        const layoutIdRaw = props && typeof props === 'object' ? (props as Record<string, unknown>).layout_id : undefined;
        const layoutId = typeof layoutIdRaw === 'number' ? layoutIdRaw : Number(layoutIdRaw ?? NaN);
        return Number.isFinite(layoutId) && selectedLayouts.includes(layoutId);
      })
      : masksData.features;

    relevantFeatures.forEach((f) => {
      const props = f.properties;
      const chunkIdRaw = props && typeof props === 'object' ? (props as Record<string, unknown>).chunk_id : undefined;
      const chunkId = typeof chunkIdRaw === 'number' ? chunkIdRaw : Number(chunkIdRaw ?? NaN);
      if (Number.isFinite(chunkId)) chunks.add(chunkId);
    });

    return Array.from(chunks).sort((a, b) => a - b).map(id => ({ label: `Chunk ${id}`, value: id }));
  }, [masksData, selectedLayouts]);

  // Handle selectedLayouts change to update selectedChunks if needed
  const handleLayoutsChange = (newLayouts: number[]) => {
    setSelectedLayouts(newLayouts);
    // If we have selected chunks, verify they still exist in the new layout selection
    // Or simpler: just clear chunks when layouts change to avoid confusion, or keep valid ones
    if (selectedChunks.length > 0 && masksData) {
      // We need to check which chunks are valid for the new layouts
      const validChunks = new Set<number>();
      const relevantFeatures = newLayouts.length > 0
        ? masksData.features.filter((f) => {
          const props = f.properties;
          const layoutIdRaw = props && typeof props === 'object' ? (props as Record<string, unknown>).layout_id : undefined;
          const layoutId = typeof layoutIdRaw === 'number' ? layoutIdRaw : Number(layoutIdRaw ?? NaN);
          return Number.isFinite(layoutId) && newLayouts.includes(layoutId);
        })
        : masksData.features;
      relevantFeatures.forEach((f) => {
        const props = f.properties;
        const chunkIdRaw = props && typeof props === 'object' ? (props as Record<string, unknown>).chunk_id : undefined;
        const chunkId = typeof chunkIdRaw === 'number' ? chunkIdRaw : Number(chunkIdRaw ?? NaN);
        if (Number.isFinite(chunkId)) validChunks.add(chunkId);
      });

      const newSelectedChunks = selectedChunks.filter(c => validChunks.has(c));
      if (newSelectedChunks.length !== selectedChunks.length) {
        setSelectedChunks(newSelectedChunks);
      }
    }
  };

  // Filter masksData
  const filteredMasksData = useMemo(() => {
    if (!masksData) return null;
    if (selectedLayouts.length === 0 && selectedChunks.length === 0) return masksData;

    return {
      ...masksData,
      features: masksData.features.filter((f) => {
        const props = f.properties;
        const layoutIdRaw = props && typeof props === 'object' ? (props as Record<string, unknown>).layout_id : undefined;
        const chunkIdRaw = props && typeof props === 'object' ? (props as Record<string, unknown>).chunk_id : undefined;
        const layoutId = typeof layoutIdRaw === 'number' ? layoutIdRaw : Number(layoutIdRaw ?? NaN);
        const chunkId = typeof chunkIdRaw === 'number' ? chunkIdRaw : Number(chunkIdRaw ?? NaN);
        const layoutMatch = selectedLayouts.length === 0 || (Number.isFinite(layoutId) && selectedLayouts.includes(layoutId));
        const chunkMatch = selectedChunks.length === 0 || (Number.isFinite(chunkId) && selectedChunks.includes(chunkId));
        return layoutMatch && chunkMatch;
      })
    };
  }, [masksData, selectedLayouts, selectedChunks]);

  const normalizeSeverity = (severity: unknown): 'high' | 'medium' | 'low' => {
    const s = String(severity ?? '').toLowerCase();
    if (s === 'high') return 'high';
    if (s === 'medium') return 'medium';
    return 'low';
  };

  const filteredDefects = useMemo(() => {
    if (defectSeverityFilter === 'all') return agentDefects;
    return agentDefects.filter((d) => normalizeSeverity(d.severity) === defectSeverityFilter);
  }, [agentDefects, defectSeverityFilter]);

  const defectMaskKeySet = useMemo(() => {
    const set = new Set<string>();
    for (const d of filteredDefects) {
      if (d.layout_id === undefined || d.chunk_id === undefined) continue;
      set.add(`${d.layout_id}:${d.chunk_id}`);
    }
    return set;
  }, [filteredDefects]);

  const maskRiskMap = useMemo(() => {
    const rank = (sev: 'high' | 'medium' | 'low') => {
      if (sev === 'high') return 3;
      if (sev === 'medium') return 2;
      return 1;
    };
    const map: Record<string, 'high' | 'medium' | 'low'> = {};
    for (const d of filteredDefects) {
      if (d.layout_id === undefined || d.chunk_id === undefined) continue;
      const key = `${d.layout_id}:${d.chunk_id}`;
      const sev = normalizeSeverity(d.severity);
      const prev = map[key];
      if (!prev || rank(sev) > rank(prev)) {
        map[key] = sev;
      }
    }
    return map;
  }, [filteredDefects]);

  const defectFilteredMasksData = useMemo(() => {
    if (!filteredMasksData) return null;
    if (defectSeverityFilter === 'all') return filteredMasksData;
    if (defectMaskKeySet.size === 0) return { ...filteredMasksData, features: [] };

    return {
      ...filteredMasksData,
      features: filteredMasksData.features.filter((f) => {
        const props = f.properties;
        const layoutIdRaw = props && typeof props === 'object' ? (props as Record<string, unknown>).layout_id : undefined;
        const chunkIdRaw = props && typeof props === 'object' ? (props as Record<string, unknown>).chunk_id : undefined;
        if (layoutIdRaw === undefined || chunkIdRaw === undefined) return false;
        return defectMaskKeySet.has(`${layoutIdRaw}:${chunkIdRaw}`);
      })
    };
  }, [filteredMasksData, defectMaskKeySet, defectSeverityFilter]);

  const processMasksToGeoJSON = useCallback((masks: unknown): FeatureCollectionLike | null => {
    // Handle new nested format { layouts: { ... }, metadata: ... }
    if (typeof masks === 'object' && masks !== null && 'layouts' in masks) {
      const layoutsValue = (masks as { layouts: unknown }).layouts;
      if (typeof layoutsValue !== 'object' || layoutsValue === null) return null;
      const features: FeatureLike[] = [];

      Object.entries(layoutsValue as Record<string, unknown>).forEach(([layoutId, layoutData]) => {
        if (typeof layoutData !== 'object' || layoutData === null) return;
        if (!('chunks' in layoutData)) return;
        const chunksValue = (layoutData as { chunks: unknown }).chunks;
        if (typeof chunksValue !== 'object' || chunksValue === null) return;

        Object.entries(chunksValue as Record<string, unknown>).forEach(([chunkId, chunkData]) => {
          // chunkData.polygons is Array<Array<[x, y]>> (List of rings)
          // We create a Feature for each polygon/ring
          if (typeof chunkData !== 'object' || chunkData === null) return;
          const polygonsValue = (chunkData as { polygons?: unknown }).polygons;
          const bboxValue = (chunkData as { bounding_box?: unknown }).bounding_box;
          if (Array.isArray(polygonsValue)) {
            polygonsValue.forEach((polygon, index) => {
              if (!Array.isArray(polygon)) return;
              features.push({
                type: "Feature",
                id: `${layoutId}-${chunkId}-${index}`,
                properties: {
                  layout_id: parseInt(layoutId),
                  chunk_id: parseInt(chunkId),
                  bbox: bboxValue
                },
                geometry: {
                  type: "Polygon",
                  // GeoJSON Polygon coordinates are [outerRing, innerRing, ...]
                  // We assume each 'polygon' in the list is a single outer ring
                  coordinates: [polygon]
                }
              });
            });
          }
        });
      });

      return {
        type: "FeatureCollection",
        features
      };
    }

    // Handle old array format
    if (!Array.isArray(masks)) {
      console.warn('Masks data is not an array or valid object:', masks);
      return null;
    }

    return {
      type: "FeatureCollection",
      features: masks.map((mask, index) => ({
        type: "Feature",
        id: index,
        properties: typeof mask === 'object' && mask !== null
          ? { ...(mask as Record<string, unknown>), segmentation: undefined, segmentation_dxf: undefined }
          : { value: mask, segmentation: undefined, segmentation_dxf: undefined },
        geometry: (() => {
          const raw =
            typeof mask === 'object' && mask !== null && ('segmentation_dxf' in mask || 'segmentation' in mask)
              ? ((mask as { segmentation_dxf?: unknown; segmentation?: unknown }).segmentation_dxf ??
                (mask as { segmentation_dxf?: unknown; segmentation?: unknown }).segmentation)
              : null;

          if (!Array.isArray(raw)) {
            return { type: "Polygon", coordinates: [] };
          }

          const isRing = raw.every((pt) =>
            Array.isArray(pt) &&
            pt.length >= 2 &&
            typeof pt[0] === 'number' &&
            typeof pt[1] === 'number'
          );
          if (!isRing) {
            return { type: "Polygon", coordinates: [] };
          }

          return {
            type: "Polygon",
            coordinates: [raw as unknown as [number, number][]]
          };
        })()
      }))
    };
  }, []);

  const processGeoJSON = useCallback((json: unknown) => {
    // Basic validation
    if (typeof json !== 'object' || json === null || !('features' in json) || !Array.isArray((json as { features: unknown }).features)) {
      alert('Invalid GeoJSON: Missing "features" array');
      setIsLoading(false);
      return;
    }
    const collection = json as FeatureCollectionLike;

    // Calculate stats
    const layouts = new Set<number>();
    const chunks = new Set<number>();
    collection.features.forEach((f) => {
      const props = f.properties;
      const layoutIdRaw = props && typeof props === 'object' ? (props as Record<string, unknown>).layout_id : undefined;
      const chunkIdRaw = props && typeof props === 'object' ? (props as Record<string, unknown>).chunk_id : undefined;
      const layoutId = typeof layoutIdRaw === 'number' ? layoutIdRaw : Number(layoutIdRaw ?? NaN);
      const chunkId = typeof chunkIdRaw === 'number' ? chunkIdRaw : Number(chunkIdRaw ?? NaN);
      if (Number.isFinite(layoutId)) layouts.add(layoutId);
      if (Number.isFinite(chunkId)) chunks.add(chunkId);
    });

    setMeta({
      layouts: layouts.size,
      chunks: chunks.size
    });

    const newBounds = calculateBounds(collection.features);
    setBounds(newBounds);
    setData(collection);

    // Pre-calculate Gallery View
    const { features: galleryFeatures, bounds: gBounds } = generateGalleryLayout(collection.features);
    setGalleryData({ ...collection, features: galleryFeatures });
    setGalleryBounds(gBounds);

    // Pre-calculate Graph View
    const gData = generateGraphData(collection.features);
    setGraphData(gData);
    setIsLoading(false);
  }, []);

  const resetDxfState = useCallback(() => {
    setSelectedFeature(null);
    setViewMode('overlay');
    setDxfId(null);
    setMasksData(null);
    setData(null);
    setGraphData(null);
    setGalleryData(null);
  }, []);

  const ensurePdfAvailable = useCallback(async () => {
    const pdfs = availablePdfs.length > 0 ? availablePdfs : await refreshPdfList(null, null);
    if (pdfs.length === 0) {
      setIsLoading(false);
      setFileName(null);
      setLoadingMessage('Please upload a PDF file for analysis.');
      alert('Please upload a PDF file first. DXF analysis requires at least one PDF to be available for analysis.');
      return false;
    }
    return true;
  }, [availablePdfs, refreshPdfList]);

  const loadDxfById = useCallback(async (id: string, displayName?: string | null) => {
    const uploadSeq = uploadSeqRef.current + 1;
    uploadSeqRef.current = uploadSeq;
    const isStale = () => uploadSeqRef.current !== uploadSeq;

    setIsLoading(true);
    setLoadingMessage('Loading CAD file...');
    setFileName(displayName ?? null);
    resetDxfState();
    setDxfId(id);

    try {
      let status = 'processing';
      const statusRes = await fetch(API_ENDPOINTS.GET_STATUS(id));
      if (!statusRes.ok) {
        throw new Error('Failed to check status');
      }
      const statusData = await statusRes.json();
      status = statusData.status;

      if (status === 'failed') {
        throw new Error(statusData.message || 'Processing failed on server');
      }

      if (status === 'processing') {
        setLoadingMessage('Processing Geometry...');
        let attempts = 0;
        const maxAttempts = 3600;

        while (status === 'processing' && attempts < maxAttempts) {
          await new Promise(resolve => setTimeout(resolve, 1000));
          if (isStale()) return;

          const pollRes = await fetch(API_ENDPOINTS.GET_STATUS(id));
          if (!pollRes.ok) {
            throw new Error('Failed to check status');
          }

          const pollData = await pollRes.json();
          status = pollData.status;
          attempts++;

          if (status === 'failed') {
            throw new Error(pollData.message || 'Processing failed on server');
          }
          if (status === 'completed') {
            break;
          }
        }

        if (status !== 'completed') {
          throw new Error('Processing timed out or did not complete');
        }
      }

      if (isStale()) return;
      const geoJsonRes = await fetch(API_ENDPOINTS.GET_DXF(id));
      if (!geoJsonRes.ok) {
        throw new Error('Failed to fetch processed GeoJSON');
      }

      const json = await geoJsonRes.json();
      processGeoJSON(json);

      try {
        if (isStale()) return;
        const masksRes = await fetch(API_ENDPOINTS.GET_MASKS(id));
        if (masksRes.ok) {
          const masksJson = await masksRes.json();
          const geoJson = processMasksToGeoJSON(masksJson);
          if (geoJson) {
            setMasksData(geoJson);
          }
        } else {
          console.warn('Failed to fetch masks:', await masksRes.text());
        }
      } catch (maskErr) {
        console.warn('Error fetching masks:', maskErr);
      }

    } catch (error) {
      console.error('DXF processing error:', error);
      console.error('API endpoint:', API_ENDPOINTS.UPLOAD_DXF);

      let errorMessage = 'Unknown error';
      if (error instanceof TypeError && error.message.includes('fetch')) {
        errorMessage = `Failed to connect to API at ${API_ENDPOINTS.UPLOAD_DXF}. Please ensure the backend server is running on port 8000.`;
      } else if (error instanceof Error) {
        errorMessage = error.message;
      }

      alert(`Error processing DXF: ${errorMessage}`);
      setIsLoading(false);
    }
  }, [processGeoJSON, processMasksToGeoJSON, resetDxfState]);

  const enrichDxfItems = useCallback(async (items: typeof availableDxfs) => {
    const processing = items.filter((dxf) => dxf.status === 'processing');
    if (processing.length === 0) return items;

    const detailed = await Promise.all(
      processing.map(async (dxf) => {
        try {
          const statusRes = await fetch(API_ENDPOINTS.GET_STATUS(dxf.dxf_id));
          if (!statusRes.ok) return dxf;
          const statusData = await statusRes.json();
          const status = typeof statusData?.status === 'string' ? statusData.status : dxf.status;
          const details = statusData?.details && typeof statusData.details === 'object'
            ? (statusData.details as { step?: string; progress?: string; error?: string })
            : undefined;
          return { ...dxf, status, status_details: details };
        } catch {
          return dxf;
        }
      })
    );

    const map = new Map(detailed.map((dxf) => [dxf.dxf_id, dxf]));
    return items.map((dxf) => map.get(dxf.dxf_id) ?? dxf);
  }, []);

  const refreshDxfList = useCallback(async (
    preferredDxfId?: string | null,
    fallbackDxfId?: string | null,
    options?: { select?: boolean }
  ) => {
    try {
      const res = await fetch(API_ENDPOINTS.LIST_DXFS);
      if (!res.ok) return [];
      const data = await res.json();
      const dxfs = (data?.dxfs || []) as typeof availableDxfs;

      const pending = availableDxfsRef.current.filter(
        (item) => item.status === 'processing' && !dxfs.some((dxf) => dxf.dxf_id === item.dxf_id)
      );

      const enriched = await enrichDxfItems([...pending, ...dxfs]);
      setAvailableDxfs(enriched);

      if (options && options.select === false) {
        return enriched;
      }

      const stored = (() => {
        try {
          return localStorage.getItem('selectedDxfId');
        } catch {
          return null;
        }
      })();

      const candidate = preferredDxfId || stored || fallbackDxfId;
      if (candidate && enriched.some((dxf) => dxf.dxf_id === candidate)) {
        if (candidate !== dxfId) {
          const match = enriched.find((dxf) => dxf.dxf_id === candidate);
          const label = match?.source_filename || match?.filename || null;
          await loadDxfById(candidate, label);
        }
        return enriched;
      }
      if (enriched.length > 0 && enriched[0].dxf_id !== dxfId) {
        const label = enriched[0].source_filename || enriched[0].filename || null;
        await loadDxfById(enriched[0].dxf_id, label);
      }
      return enriched;
    } catch {
      return [];
    }
  }, [dxfId, loadDxfById, enrichDxfItems]);

  useEffect(() => {
    refreshDxfList(null, null);
  }, [refreshDxfList]);

  useEffect(() => {
    if (availableDxfs.every((dxf) => dxf.status !== 'processing')) return;
    const timer = window.setInterval(() => {
      refreshDxfList(null, null, { select: false });
    }, 2000);

    return () => window.clearInterval(timer);
  }, [availableDxfs, refreshDxfList]);

  const uploadPdfFile = async (file: File) => {
    setIsLoading(true);
    setLoadingMessage('Processing file...');
    setFileName(file.name);
    setPdfId(null);
    setShowPdfPanel(false);

    try {
      const formData = new FormData();
      formData.append('file', file);

      setLoadingMessage('Uploading PDF...');
      const uploadRes = await fetch(API_ENDPOINTS.UPLOAD_PDF, {
        method: 'POST',
        body: formData,
      });

      if (!uploadRes.ok) {
        const err = await uploadRes.json();
        throw new Error(err.detail || 'Upload failed');
      }

      const uploadData = await uploadRes.json();
      const id = uploadData?.id as string | undefined;
      const uploadStatus = typeof uploadData?.status === 'string' ? uploadData.status : 'processing';
      const sourceName = typeof uploadData?.pdf_source === 'string' ? uploadData.pdf_source : file.name;
      if (!id) {
        throw new Error('Upload response missing file id');
      }

      setAvailablePdfs((prev) => {
        if (prev.some((pdf) => pdf.pdf_id === id)) return prev;
        return [
          {
            pdf_id: id,
            filename: file.name,
            pdf_source: sourceName,
            status: uploadStatus
          },
          ...prev
        ];
      });

      let status = 'processing';
      let attempts = 0;
      const maxAttempts = 3600;

      while (status === 'processing' && attempts < maxAttempts) {
        await new Promise(resolve => setTimeout(resolve, 1000));

        const statusRes = await fetch(API_ENDPOINTS.GET_PDF_STATUS(id));
        if (!statusRes.ok) throw new Error('Failed to check status');

        const statusData = await statusRes.json();
        status = statusData.status;
        attempts++;

        if (statusData.details?.step) {
          const step = statusData.details.step.replace('_', ' ');
          const progress = statusData.details.progress ? ` (${statusData.details.progress})` : '';
          setLoadingMessage(`Processing: ${step}${progress}...`);
        }
        if (statusData.details && typeof statusData.details === 'object') {
          const details = statusData.details as PdfStatusDetails;
          setAvailablePdfs((prev) =>
            prev.map((pdf) =>
              pdf.pdf_id === id ? { ...pdf, status, status_details: details } : pdf
            )
          );
        }

        if (status === 'failed') {
          throw new Error(statusData.details?.error || 'Processing failed');
        }
      }

      if (status !== 'completed') {
        throw new Error('Processing timed out');
      }

      setPdfId(id);
      setShowPdfPanel(true);
      refreshPdfList(id, id);
      setIsLoading(false);
    } catch (error) {
      console.error('PDF processing error:', error);
      alert(`Error processing PDF: ${error instanceof Error ? error.message : 'Unknown error'}`);
      setIsLoading(false);
    }
  };

  const handlePdfUpload = async (files: FileList) => {
    const list = Array.from(files);
    for (const file of list) {
      await uploadPdfFile(file);
    }
  };

  const uploadDxfFile = async (file: File) => {
    const formData = new FormData();
    formData.append('file', file);

    const uploadRes = await fetch(API_ENDPOINTS.UPLOAD_DXF, {
      method: 'POST',
      body: formData,
    });

    if (!uploadRes.ok) {
      const err = await uploadRes.json();
      throw new Error(err.detail || 'Upload failed');
    }

    const uploadDataUnknown: unknown = await uploadRes.json();

    if (!uploadDataUnknown || typeof uploadDataUnknown !== 'object') {
      throw new Error('Invalid upload response from server');
    }

    const uploadData = uploadDataUnknown as Record<string, unknown>;
    const id = uploadData.id;
    if (typeof id !== 'string' || !id) {
      throw new Error('Upload response missing file id');
    }

    return id;
  };

  const handleDxfUpload = async (files: FileList) => {
    const list = Array.from(files);
    if (list.length === 0) return;

    let lastUploadedId: string | null = null;
    for (let i = 0; i < list.length; i += 1) {
      const file = list[i];
      const isLast = i === list.length - 1;

      try {
        setIsLoading(true);
        setLoadingMessage('Uploading DXF...');
        setFileName(file.name);
        resetDxfState();
        const id = await uploadDxfFile(file);
        lastUploadedId = id;
        setAvailableDxfs((prev) => {
          if (prev.some((dxf) => dxf.dxf_id === id)) return prev;
          return [
            {
              dxf_id: id,
              filename: file.name,
              source_filename: file.name,
              status: 'processing',
              status_details: { step: 'uploading' }
            },
            ...prev
          ];
        });
        if (isLast) {
          await loadDxfById(id, file.name);
        } else {
          setIsLoading(false);
        }
      } catch (error) {
        console.error('DXF processing error:', error);
        console.error('API endpoint:', API_ENDPOINTS.UPLOAD_DXF);

        let errorMessage = 'Unknown error';
        if (error instanceof TypeError && error.message.includes('fetch')) {
          errorMessage = `Failed to connect to API at ${API_ENDPOINTS.UPLOAD_DXF}. Please ensure the backend server is running on port 8000.`;
        } else if (error instanceof Error) {
          errorMessage = error.message;
        }

        alert(`Error processing DXF: ${errorMessage}`);
        setIsLoading(false);
        return;
      }
    }

    refreshDxfList(lastUploadedId, lastUploadedId);
  };

  const handleDeletePdf = async (id: string) => {
    const ok = window.confirm('Delete this PDF?');
    if (!ok) return;
    try {
      const res = await fetch(API_ENDPOINTS.DELETE_PDF(id), { method: 'DELETE' });
      if (!res.ok) throw new Error('Failed to delete PDF');
      setAvailablePdfs((prev) => {
        const next = prev.filter((p) => p.pdf_id !== id);
        if (pdfId === id) {
          const nextId = next.length > 0 ? next[0].pdf_id : null;
          setPdfId(nextId);
          if (!nextId) setShowPdfPanel(false);
        }
        return next;
      });
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Failed to delete PDF';
      alert(msg);
    }
  };

  const handleDeleteDxf = async (id: string) => {
    const ok = window.confirm('Delete this DXF?');
    if (!ok) return;
    try {
      const res = await fetch(API_ENDPOINTS.DELETE_DXF(id), { method: 'DELETE' });
      if (!res.ok) {
        const err = await res.json().catch(() => ({} as unknown));
        const msg = typeof (err as Record<string, unknown>).detail === 'string' ? (err as Record<string, string>).detail : 'Failed to delete DXF';
        throw new Error(msg);
      }
      setAvailableDxfs((prev) => {
        const next = prev.filter((d) => d.dxf_id !== id);
        if (dxfId === id) {
          resetDxfState();
          setFileName(null);
        }
        return next;
      });
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Failed to delete DXF';
      alert(msg);
    }
  };

  const handleSelectPdfId = (id: string) => {
    setPdfId(id);
    setShowPdfPanel(true);
  };

  const handleSelectDxfId = async (id: string) => {
    const match = availableDxfs.find((dxf) => dxf.dxf_id === id);
    const label = match?.source_filename || match?.filename || null;
    await loadDxfById(id, label);
  };

  const handleDefectSelect = (defect: Defect | null) => {
    if (!defect) {
      const props = selectedFeature?.properties;
      const isDefect =
        props && typeof props === 'object'
          ? (props as Record<string, unknown>).isDefect === true
          : false;
      if (selectedFeature && isDefect) {
        setSelectedFeature(null);
      }
      // Always clear filters when deselecting a defect to ensure full view is restored
      setSelectedLayouts([]);
      setSelectedChunks([]);
      return;
    }

    // Prepare new filters based on the selected defect
    const nextLayouts: number[] = [];
    const nextChunks: number[] = [];

    if (defect.layout_id !== undefined && defect.layout_id !== null) {
      const lId = Number(defect.layout_id);
      if (Number.isFinite(lId)) {
        nextLayouts.push(lId);
      }
    }
    
    if (defect.chunk_id !== undefined && defect.chunk_id !== null) {
      const cId = Number(defect.chunk_id);
      if (Number.isFinite(cId)) {
        nextChunks.push(cId);
      }
    }

    // Set filters directly (overwriting any previous state)
    setSelectedLayouts(nextLayouts);
    setSelectedChunks(nextChunks);

    // Ensure the defect is visible by resetting severity filter if needed
    // We'll just reset to 'all' to be safe and ensure the selected defect is shown
    setDefectSeverityFilter('all');

    // Create a feature-like object for highlighting
    setSelectedFeature({
      type: "Feature",
      properties: {
        layout_id: defect.layout_id,
        chunk_id: defect.chunk_id,
        isDefect: true
      },
      geometry: { type: 'Point', coordinates: [0, 0] }
    });
  };

  const handleRunAgentAnalysis = async () => {
    if (!dxfId) {
      alert('Please upload a DXF file first.');
      return;
    }
    const canProceed = await ensurePdfAvailable();
    if (!canProceed) return;
    setAgentDefects([]);
    setDefectSeverityFilter('all');
    setRibbonTab('analysis');
    setShowAgentPanel(true);
    setAgentRunId((prev) => prev + 1);
  };

  const handleSeverityFilterChange = (next: 'all' | 'high' | 'medium' | 'low') => {
    setDefectSeverityFilter(next);
    if (next !== 'all') {
      setViewMode('raster');
    }
  };

  return (
    <div className="flex flex-col h-screen w-screen overflow-hidden bg-cad-bg">
      <TopNav
        viewMode={viewMode}
        onViewModeChange={setViewMode}
        ribbonTab={ribbonTab}
        onRibbonTabChange={setRibbonTab}
        agentQuery={agentQuery}
        onAgentQueryChange={setAgentQuery}
        onRunAgentAnalysis={handleRunAgentAnalysis}
        fileName={fileName}
        layoutOptions={layoutOptions}
        chunkOptions={chunkOptions}
        selectedLayouts={selectedLayouts}
        onLayoutsChange={handleLayoutsChange}
        selectedChunks={selectedChunks}
        onChunksChange={setSelectedChunks}
        hasPdf={availablePdfs.length > 0}
        showPdfPanel={showPdfPanel}
        onTogglePdfPanel={setShowPdfPanel}
        onToggleAgentPanel={setShowAgentPanel}
        showAgentPanel={showAgentPanel}
        onToggleChatPanel={setShowChatPanel}
        showChatPanel={showChatPanel}
      />
      <div className="flex flex-1 overflow-hidden relative">
        <ControlPanel
          onUploadPdfs={handlePdfUpload}
          onUploadDxfs={handleDxfUpload}
          onSelectPdfId={handleSelectPdfId}
          onSelectDxfId={handleSelectDxfId}
          onDeletePdf={handleDeletePdf}
          onDeleteDxf={handleDeleteDxf}
          pdfs={availablePdfs}
          dxfs={availableDxfs}
          selectedPdfId={pdfId}
          selectedDxfId={dxfId}
          fileName={fileName}
          meta={meta}
          selectedFeature={selectedFeature}
          viewMode={viewMode}
          onViewModeChange={setViewMode}
          showMasks={showMasks}
          onToggleMasks={setShowMasks}
          maskColorMode={maskColorMode}
          onMaskColorModeChange={setMaskColorMode}
        />

        <div className="flex-1 relative">
          {isLoading && (
            <div className="fixed inset-0 z-[1000] flex items-center justify-center bg-black/70 backdrop-blur-md">
              <div className="flex flex-col items-center gap-5 px-10 py-8 bg-[#121418] rounded-2xl border border-white/10 shadow-2xl animate-in fade-in zoom-in duration-200 min-w-[320px] max-w-[560px] mx-6">
                <div className="relative">
                  <div className="absolute inset-0 bg-blue-500/20 blur-xl rounded-full"></div>
                  <Loader2 className="relative w-14 h-14 animate-spin text-blue-500" />
                </div>
                <div className="flex flex-col items-center gap-2 text-center">
                  <div className="text-xl font-semibold text-white">Processing</div>
                  {fileName && <div className="text-sm text-neutral-300 truncate max-w-[480px]">{fileName}</div>}
                  <div className="text-sm text-neutral-400">{loadingMessage}</div>
                </div>
              </div>
            </div>
          )}
          <Viewer
            data={viewMode === 'gallery' ? galleryData : data}
            graphData={viewMode === 'graph' ? graphData : null}
            bounds={viewMode === 'gallery' ? galleryBounds : bounds}
            onSelect={setSelectedFeature}
            selectedFeature={selectedFeature}
            viewMode={viewMode}
            dxfId={dxfId}
            masksData={defectFilteredMasksData}
            showMasks={showMasks}
            maskColorMode={maskColorMode}
            maskRiskMap={maskRiskMap}
          />
        </div>

        {showPdfPanel && (
          <PdfPanel
            pdfId={pdfId}
            pdfs={availablePdfs}
            onSelectPdfId={(next) => {
              setPdfId(next);
              setShowPdfPanel(true);
            }}
            onClose={() => setShowPdfPanel(false)}
          />
        )}

        {showAgentPanel && dxfId && (
          <div className={ribbonTab === 'analysis' ? '' : 'invisible pointer-events-none'}>
            <AgentPanel
              fileId={dxfId}
              query={agentQuery}
              runId={agentRunId}
              severityFilter={defectSeverityFilter}
              onSeverityFilterChange={handleSeverityFilterChange}
              onDefectsChange={setAgentDefects}
              onClose={() => setShowAgentPanel(false)}
              onDefectSelect={handleDefectSelect}
            />
          </div>
        )}
        {showChatPanel && (
          <ChatPanel 
            fileName={fileName} 
            onClose={() => setShowChatPanel(false)} 
            onDefectSelect={handleDefectSelect}
          />
        )}
      </div>
    </div>
  );
}

export default App;
