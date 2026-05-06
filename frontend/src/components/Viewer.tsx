import React, { useState, useEffect, useCallback } from 'react';
import DeckGL from '@deck.gl/react';
import { GeoJsonLayer, ScatterplotLayer, LineLayer, BitmapLayer } from '@deck.gl/layers';
import { OrthographicView, OrthographicViewport, COORDINATE_SYSTEM } from '@deck.gl/core';
import { getChunkColor } from '../utils/geoUtils';
import { Plus, Minus, Maximize } from 'lucide-react';
import type { Bounds, FeatureCollectionLike, FeatureLike, GraphData, GraphEdge, GraphNode } from '../utils/geoUtils';
import { API_ENDPOINTS } from '../config/api';
import type { Layer, PickingInfo, OrthographicViewState, ViewStateChangeParameters } from '@deck.gl/core';

interface ViewerProps {
  data: FeatureCollectionLike | null;
  bounds: Bounds | null;
  onSelect?: (feature: FeatureLike) => void;
  selectedFeature?: FeatureLike | null;
  viewMode?: 'overlay' | 'gallery' | 'graph' | 'raster';
  graphData?: GraphData | null;
  dxfId?: string | null;
  masksData?: FeatureCollectionLike | null;
  showMasks?: boolean;
  maskColorMode?: 'single' | 'distinct';
  maskRiskMap?: Record<string, 'high' | 'medium' | 'low'>;
}

interface ImageTransform {
  min_x: number;
  max_x: number;
  min_y: number;
  max_y: number;
  image_width: number;
  image_height: number;
}

const INITIAL_VIEW_STATE: { target: [number, number, number], zoom: number, minZoom: number, maxZoom: number } = {
  target: [0, 0, 0],
  zoom: 0,
  minZoom: -20,
  maxZoom: 20
};

export default function Viewer({ data, bounds, onSelect, selectedFeature, viewMode = 'overlay', graphData, dxfId, masksData, showMasks = true, maskColorMode = 'single', maskRiskMap }: ViewerProps) {
  const [viewState, setViewState] = useState(INITIAL_VIEW_STATE);
  const [imageTransform, setImageTransform] = useState<ImageTransform | null>(null);
  const [dimensions, setDimensions] = useState({ width: 1, height: 1 });
  const containerRef = React.useRef<HTMLDivElement>(null);

  const getProp = (props: FeatureLike['properties'], key: string): unknown => {
    if (!props || typeof props !== 'object') return undefined;
    return (props as Record<string, unknown>)[key];
  };

  const emptyCollection: FeatureCollectionLike = { type: 'FeatureCollection', features: [] };

  const getRiskColor = (severity: 'high' | 'medium' | 'low', alpha: number): [number, number, number, number] => {
    if (severity === 'high') return [239, 68, 68, alpha];
    if (severity === 'medium') return [234, 179, 8, alpha];
    return [59, 130, 246, alpha];
  };

  const getMaskKey = (props: FeatureLike['properties']): string | null => {
    const layoutId = getProp(props, 'layout_id');
    const chunkId = getProp(props, 'chunk_id');
    if (layoutId === undefined || chunkId === undefined) return null;
    return `${layoutId}:${chunkId}`;
  };

  useEffect(() => {
    if (!containerRef.current) return;
    const obs = new ResizeObserver(entries => {
      const { width, height } = entries[0].contentRect;
      setDimensions({ width, height });
    });
    obs.observe(containerRef.current);
    return () => obs.disconnect();
  }, []);

  useEffect(() => {
    if (viewMode !== 'raster' || !dxfId) {
      setImageTransform(null);
      return;
    }

    const controller = new AbortController();
    const load = async () => {
      try {
        const res = await fetch(API_ENDPOINTS.GET_IMAGE_TRANSFORM(dxfId), { signal: controller.signal });
        if (!res.ok) {
          setImageTransform(null);
          return;
        }
        const json: unknown = await res.json();
        if (
          typeof json === 'object' &&
          json !== null &&
          typeof (json as Record<string, unknown>).min_x === 'number' &&
          typeof (json as Record<string, unknown>).max_x === 'number' &&
          typeof (json as Record<string, unknown>).min_y === 'number' &&
          typeof (json as Record<string, unknown>).max_y === 'number' &&
          typeof (json as Record<string, unknown>).image_width === 'number' &&
          typeof (json as Record<string, unknown>).image_height === 'number'
        ) {
          setImageTransform(json as ImageTransform);
          return;
        }
        setImageTransform(null);
      } catch {
        if (controller.signal.aborted) return;
        setImageTransform(null);
      }
    };

    void load();
    return () => controller.abort();
  }, [dxfId, viewMode]);

  const fitBounds = useCallback(() => {
    const effectiveBounds: Bounds | null =
      viewMode === 'raster' && imageTransform
        ? (() => {
          const minX = imageTransform.min_x;
          const maxX = imageTransform.max_x;
          const minY = imageTransform.min_y;
          const maxY = imageTransform.max_y;
          const width = maxX - minX;
          const height = maxY - minY;
          const centerX = minX + width / 2;
          const centerY = minY + height / 2;
          return { minX, minY, maxX, maxY, width, height, centerX, centerY };
        })()
        : bounds;

    if (effectiveBounds && containerRef.current) {
      const { width, height, centerX, centerY } = effectiveBounds;
      if (width === 0 || height === 0) return;

      const containerWidth = containerRef.current.clientWidth;
      const containerHeight = containerRef.current.clientHeight;
      const padding = 1.1;

      const zoomX = Math.log2(containerWidth / (width * padding));
      const zoomY = Math.log2(containerHeight / (height * padding));
      const zoom = Math.min(zoomX, zoomY);

      setViewState({
        ...INITIAL_VIEW_STATE,
        target: [centerX, centerY, 0],
        zoom: zoom
      });
    }
  }, [bounds, imageTransform, viewMode]);

  // Auto-fit when bounds change
  useEffect(() => {
    fitBounds();
  }, [fitBounds]);

  const handleZoomIn = () => {
    setViewState(v => ({ ...v, zoom: Math.min(v.zoom + 1, v.maxZoom) }));
  };

  const handleZoomOut = () => {
    setViewState(v => ({ ...v, zoom: Math.max(v.zoom - 1, v.minZoom) }));
  };

  const layers: Layer[] = [];

  if (viewMode === 'graph' && graphData) {
    layers.push(
      new LineLayer<GraphEdge>({
        id: 'graph-edges',
        data: graphData.edges,
        pickable: false,
        getWidth: 1,
        getSourcePosition: (d) => d.sourcePosition,
        getTargetPosition: (d) => d.targetPosition,
        getColor: (d) => d.color,
        coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
        opacity: 0.5,
      }),
      new ScatterplotLayer<GraphNode>({
        id: 'graph-nodes',
        data: graphData.nodes,
        pickable: true,
        opacity: 1,
        stroked: true,
        filled: true,
        radiusScale: 1,
        radiusMinPixels: 4,
        radiusMaxPixels: 15,
        lineWidthMinPixels: 1,
        getPosition: (d) => d.position,
        getFillColor: (d) => d.color,
        coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
        onClick: (info: PickingInfo<GraphNode>) => {
          if (info.object && onSelect) {
            // Mimic the GeoJSON feature structure so ControlPanel works
            onSelect({
              type: 'Feature',
              properties: info.object.properties,
              geometry: { type: 'Point', coordinates: info.object.position }
            });
          }
        },
        updateTriggers: {
          getLineColor: [selectedFeature],
          getLineWidth: [selectedFeature]
        },
        getLineWidth: (d) => {
          const selectedNodeId = getProp(selectedFeature?.properties ?? null, 'node_id');
          const isSelected = selectedFeature && String(selectedNodeId) === String(d.id);
          return isSelected ? 2 : 1;
        },
        // Highlight selected node with a white ring
        getLineColor: (d) => {
          const selectedNodeId = getProp(selectedFeature?.properties ?? null, 'node_id');
          const isSelected = selectedFeature && String(selectedNodeId) === String(d.id);
          return isSelected ? [255, 255, 255] : [0, 0, 0];
        }
      })
    );
  } else if (viewMode === 'raster' && dxfId && (bounds || imageTransform)) {
    // SVG overlay is handled outside DeckGL layers for high resolution
    
    if (masksData && showMasks) {
      layers.push(
        new GeoJsonLayer<FeatureLike>({
          id: 'masks-layer',
          data: masksData ?? emptyCollection,
          pickable: true,
          stroked: true,
          filled: true,
          lineWidthMinPixels: 1,
          coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
          getFillColor: (f) => {
            const selectedLayoutId = getProp(selectedFeature?.properties ?? null, 'layout_id');
            const selectedChunkId = getProp(selectedFeature?.properties ?? null, 'chunk_id');
            const layoutId = getProp(f.properties ?? null, 'layout_id');
            const chunkId = getProp(f.properties ?? null, 'chunk_id');

            const isSelected = selectedFeature &&
              String(selectedLayoutId) === String(layoutId) &&
              String(selectedChunkId) === String(chunkId);

            if (isSelected) return [255, 255, 0, 180]; // Highlight selected

            const key = getMaskKey(f.properties ?? null);
            const severity = key ? maskRiskMap?.[key] : undefined;
            if (severity) return getRiskColor(severity, 140);

            if (maskColorMode === 'single') {
              return [255, 0, 0, 100];
            }
            const chunkIdRaw = getProp(f.properties ?? null, 'chunk_id') ?? 0;
            const chunkIdNum = typeof chunkIdRaw === 'number' ? chunkIdRaw : Number(chunkIdRaw);
            const color = getChunkColor(Number.isFinite(chunkIdNum) ? chunkIdNum : 0);
            return [...color, 150];
          },
          getLineColor: (f) => {
            const selectedLayoutId = getProp(selectedFeature?.properties ?? null, 'layout_id');
            const selectedChunkId = getProp(selectedFeature?.properties ?? null, 'chunk_id');
            const layoutId = getProp(f.properties ?? null, 'layout_id');
            const chunkId = getProp(f.properties ?? null, 'chunk_id');

            const isSelected = selectedFeature &&
              String(selectedLayoutId) === String(layoutId) &&
              String(selectedChunkId) === String(chunkId);

            if (isSelected) return [255, 255, 255, 255]; // White border for selected

            const key = getMaskKey(f.properties ?? null);
            const severity = key ? maskRiskMap?.[key] : undefined;
            if (severity) return getRiskColor(severity, 255);

            if (maskColorMode === 'single') {
              return [255, 0, 0, 255];
            }
            const chunkIdRaw = getProp(f.properties ?? null, 'chunk_id') ?? 0;
            const chunkIdNum = typeof chunkIdRaw === 'number' ? chunkIdRaw : Number(chunkIdRaw);
            const color = getChunkColor(Number.isFinite(chunkIdNum) ? chunkIdNum : 0);
            return [...color, 255];
          },
          getLineWidth: (f) => {
            const selectedLayoutId = getProp(selectedFeature?.properties ?? null, 'layout_id');
            const selectedChunkId = getProp(selectedFeature?.properties ?? null, 'chunk_id');
            const layoutId = getProp(f.properties ?? null, 'layout_id');
            const chunkId = getProp(f.properties ?? null, 'chunk_id');

            const isSelected = selectedFeature &&
              String(selectedLayoutId) === String(layoutId) &&
              String(selectedChunkId) === String(chunkId);
            return isSelected ? 3 : 1;
          },
          onClick: (info: PickingInfo<FeatureLike>) => {
            if (info.object && onSelect) {
              onSelect(info.object);
            }
          },
          updateTriggers: {
            getFillColor: [maskColorMode, selectedFeature, maskRiskMap],
            getLineColor: [maskColorMode, selectedFeature, maskRiskMap],
            getLineWidth: [selectedFeature]
          }
        })
      );
    }
  } else {
    layers.push(
      new GeoJsonLayer<FeatureLike>({
        id: 'geojson-layer',
        data: data ?? emptyCollection,
        pickable: true,
        stroked: true,
        filled: true,
        lineWidthMinPixels: 1,
        coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
        getFillColor: (f) => {
          const chunkIdRaw = getProp(f.properties ?? null, 'chunk_id') ?? 0;
          const chunkIdNum = typeof chunkIdRaw === 'number' ? chunkIdRaw : Number(chunkIdRaw);
          return getChunkColor(Number.isFinite(chunkIdNum) ? chunkIdNum : 0);
        },
        getLineColor: (f) => {
          const selLayoutId = getProp(selectedFeature?.properties ?? null, 'layout_id');
          const selChunkId = getProp(selectedFeature?.properties ?? null, 'chunk_id');
          const selNodeId = getProp(selectedFeature?.properties ?? null, 'node_id');
          const layoutId = getProp(f.properties ?? null, 'layout_id');
          const chunkId = getProp(f.properties ?? null, 'chunk_id');
          const nodeId = getProp(f.properties ?? null, 'node_id');

          const isSelected = selectedFeature &&
            String(selLayoutId) === String(layoutId) &&
            String(selChunkId) === String(chunkId) &&
            String(selNodeId) === String(nodeId);
          return isSelected ? [255, 255, 255, 255] : [0, 0, 0, 255];
        },
        getLineWidth: (f) => {
          const selLayoutId = getProp(selectedFeature?.properties ?? null, 'layout_id');
          const selChunkId = getProp(selectedFeature?.properties ?? null, 'chunk_id');
          const selNodeId = getProp(selectedFeature?.properties ?? null, 'node_id');
          const layoutId = getProp(f.properties ?? null, 'layout_id');
          const chunkId = getProp(f.properties ?? null, 'chunk_id');
          const nodeId = getProp(f.properties ?? null, 'node_id');

          const isSelected = selectedFeature &&
            String(selLayoutId) === String(layoutId) &&
            String(selChunkId) === String(chunkId) &&
            String(selNodeId) === String(nodeId);
          return isSelected ? 3 : 1;
        },
        opacity: 0.9,
        onClick: (info: PickingInfo<FeatureLike>) => {
          if (info.object && onSelect) {
            onSelect(info.object);
          }
        },
        updateTriggers: {
          getFillColor: [data],
          getLineColor: [selectedFeature],
          getLineWidth: [selectedFeature]
        }
      })
    );

    if (masksData && showMasks) {
      layers.push(
        new GeoJsonLayer<FeatureLike>({
          id: 'masks-layer-overlay',
          data: masksData ?? emptyCollection,
          pickable: true,
          stroked: true,
          filled: true,
          lineWidthMinPixels: 1,
          coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
          getFillColor: (f) => {
            const selectedLayoutId = getProp(selectedFeature?.properties ?? null, 'layout_id');
            const selectedChunkId = getProp(selectedFeature?.properties ?? null, 'chunk_id');
            const layoutId = getProp(f.properties ?? null, 'layout_id');
            const chunkId = getProp(f.properties ?? null, 'chunk_id');

            const isSelected = selectedFeature &&
              String(selectedLayoutId) === String(layoutId) &&
              String(selectedChunkId) === String(chunkId);

            if (isSelected) return [255, 255, 255, 30];

            const key = getMaskKey(f.properties ?? null);
            const severity = key ? maskRiskMap?.[key] : undefined;
            if (severity) return getRiskColor(severity, 80);

            if (maskColorMode === 'single') {
              return [255, 0, 0, 50];
            }
            const chunkIdRaw = getProp(f.properties ?? null, 'chunk_id') ?? 0;
            const chunkIdNum = typeof chunkIdRaw === 'number' ? chunkIdRaw : Number(chunkIdRaw);
            const color = getChunkColor(Number.isFinite(chunkIdNum) ? chunkIdNum : 0);
            return [...color, 60];
          },
          getLineColor: (f) => {
            const selectedLayoutId = getProp(selectedFeature?.properties ?? null, 'layout_id');
            const selectedChunkId = getProp(selectedFeature?.properties ?? null, 'chunk_id');
            const layoutId = getProp(f.properties ?? null, 'layout_id');
            const chunkId = getProp(f.properties ?? null, 'chunk_id');

            const isSelected = selectedFeature &&
              String(selectedLayoutId) === String(layoutId) &&
              String(selectedChunkId) === String(chunkId);

            if (isSelected) return [255, 255, 255, 255];

            const key = getMaskKey(f.properties ?? null);
            const severity = key ? maskRiskMap?.[key] : undefined;
            if (severity) return getRiskColor(severity, 255);

            if (maskColorMode === 'single') {
              return [255, 0, 0, 255];
            }
            const chunkIdRaw = getProp(f.properties ?? null, 'chunk_id') ?? 0;
            const chunkIdNum = typeof chunkIdRaw === 'number' ? chunkIdRaw : Number(chunkIdRaw);
            const color = getChunkColor(Number.isFinite(chunkIdNum) ? chunkIdNum : 0);
            return [...color, 255];
          },
          getLineWidth: (f) => {
            const selectedLayoutId = getProp(selectedFeature?.properties ?? null, 'layout_id');
            const selectedChunkId = getProp(selectedFeature?.properties ?? null, 'chunk_id');
            const layoutId = getProp(f.properties ?? null, 'layout_id');
            const chunkId = getProp(f.properties ?? null, 'chunk_id');

            const isSelected = selectedFeature &&
              String(selectedLayoutId) === String(layoutId) &&
              String(selectedChunkId) === String(chunkId);
            return isSelected ? 3 : 1;
          },
          onClick: (info: PickingInfo<FeatureLike>) => {
            if (info.object && onSelect) {
              onSelect(info.object);
            }
          },
          updateTriggers: {
            getFillColor: [maskColorMode, selectedFeature, maskRiskMap],
            getLineColor: [maskColorMode, selectedFeature, maskRiskMap],
            getLineWidth: [selectedFeature]
          }
        })
      );
    }
  }

  const svgStyle = React.useMemo(() => {
    if (!imageTransform || viewMode !== 'raster' || !dxfId) return null;

    const viewport = new OrthographicViewport({
      ...viewState,
      width: dimensions.width,
      height: dimensions.height,
      flipY: false
    });

    // Project top-left (min_x, max_y)
    const [x1, y1] = viewport.project([imageTransform.min_x, imageTransform.max_y, 0]);
    // Project bottom-right (max_x, min_y)
    const [x2, y2] = viewport.project([imageTransform.max_x, imageTransform.min_y, 0]);

    return {
      position: 'absolute' as const,
      left: x1,
      top: y1,
      width: x2 - x1,
      height: y2 - y1,
      pointerEvents: 'none' as const,
      zIndex: 0
    };
  }, [viewState, imageTransform, dimensions, viewMode, dxfId]);

  return (
    <div ref={containerRef} className="relative w-full h-full bg-[#111] overflow-hidden">
      {svgStyle && dxfId && (
        <img 
          src={API_ENDPOINTS.GET_IMAGE_CONTENT(dxfId)} 
          alt="DXF Raster"
          style={svgStyle}
          className="absolute"
        />
      )}
      <DeckGL
        viewState={viewState}
        controller={true}
        layers={layers}
        style={{ width: '100%', height: '100%' }}
        onViewStateChange={(params: ViewStateChangeParameters<OrthographicViewState>) => {
          const next = params.viewState;
          const nextTarget = Array.isArray(next.target) ? next.target : viewState.target;
          const target3: [number, number, number] = [
            Number(nextTarget[0] ?? 0),
            Number(nextTarget[1] ?? 0),
            Number((nextTarget as number[])[2] ?? 0)
          ];

          setViewState((prev) => ({
            ...prev,
            zoom: typeof next.zoom === 'number' ? next.zoom : prev.zoom,
            target: target3
          }));
        }}
        views={new OrthographicView({ controller: true, flipY: false })}
      >
      </DeckGL>

      {/* Controls */}
      <div className="absolute top-4 right-4 flex flex-col gap-2">
        <button
          onClick={handleZoomIn}
          className="p-2 bg-[#1a1d21] border border-white/10 rounded-lg text-neutral-300 hover:text-white hover:bg-white/5 transition-colors"
          title="Zoom In"
        >
          <Plus className="w-5 h-5" />
        </button>
        <button
          onClick={handleZoomOut}
          className="p-2 bg-[#1a1d21] border border-white/10 rounded-lg text-neutral-300 hover:text-white hover:bg-white/5 transition-colors"
          title="Zoom Out"
        >
          <Minus className="w-5 h-5" />
        </button>
        <button
          onClick={fitBounds}
          className="p-2 bg-[#1a1d21] border border-white/10 rounded-lg text-neutral-300 hover:text-white hover:bg-white/5 transition-colors"
          title="Fit to Bounds"
        >
          <Maximize className="w-5 h-5" />
        </button>
      </div>

      {/* Info Overlay */}
      <div className="absolute bottom-4 left-4 text-[10px] font-mono text-neutral-500 pointer-events-none select-none">
        <div>X: {viewState.target[0].toFixed(2)}</div>
        <div>Y: {viewState.target[1].toFixed(2)}</div>
        <div>Z: {viewState.zoom.toFixed(2)}</div>
      </div>
    </div>
  );
}
