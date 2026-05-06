import type { Feature, FeatureCollection, Geometry, GeoJsonProperties, Position } from 'geojson';

export const TAB20_COLORS = [
  [31, 119, 180], [174, 199, 232], [255, 127, 14], [255, 187, 120], 
  [44, 160, 44], [152, 223, 138], [214, 39, 40], [255, 152, 150], 
  [148, 103, 189], [197, 176, 213], [140, 86, 75], [196, 156, 148], 
  [227, 119, 194], [247, 182, 210], [127, 127, 127], [199, 199, 199], 
  [188, 189, 34], [219, 219, 141], [23, 190, 207], [158, 218, 229]
];

export function getChunkColor(chunkId: number): [number, number, number] {
  // Ensure we return a tuple of 3 numbers
  const color = TAB20_COLORS[chunkId % TAB20_COLORS.length];
  return [color[0], color[1], color[2]];
}

export type FeatureLike = Feature<Geometry, GeoJsonProperties>;
export type FeatureCollectionLike = FeatureCollection<Geometry, GeoJsonProperties>;

export interface Bounds {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
  width: number;
  height: number;
  centerX: number;
  centerY: number;
}

export function calculateBounds(features: FeatureLike[]): Bounds | null {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  let hasPoints = false;

  for (const feature of features) {
      const geometry = feature.geometry;
      if (!geometry) continue;

      let rings: Position[][] = [];
      if (geometry.type === 'Polygon') {
        rings = geometry.coordinates;
      } else if (geometry.type === 'MultiPolygon') {
        rings = geometry.coordinates.flat(1);
      } else {
        continue;
      }

      for (const ring of rings) {
        for (const point of ring) {
          const x = point[0];
          const y = point[1];
          if (typeof x === 'number' && typeof y === 'number') {
            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
            hasPoints = true;
          }
        }
      }
  }

  if (!hasPoints) return null;

  const width = maxX - minX;
  const height = maxY - minY;
  const centerX = minX + width / 2;
  const centerY = minY + height / 2;

  return { minX, minY, maxX, maxY, width, height, centerX, centerY };
}

export function generateGalleryLayout(features: FeatureLike[], gap: number = 50): { features: FeatureLike[], bounds: Bounds | null } {
  // 1. Group by chunk_id
  const chunks = new Map<number, FeatureLike[]>();
  features.forEach(f => {
    const props = f.properties;
    const cidRaw =
      props && typeof props === 'object'
        ? (props as Record<string, unknown>).chunk_id
        : -1;
    const cid = typeof cidRaw === 'number' ? cidRaw : Number(cidRaw ?? -1);
    if (!chunks.has(cid)) chunks.set(cid, []);
    chunks.get(cid)?.push(f);
  });

  const chunkIds = Array.from(chunks.keys()).sort((a, b) => a - b);
  const nChunks = chunkIds.length;
  if (nChunks === 0) return { features: [], bounds: null };

  // 2. Calculate bounds for each chunk and find max dimensions
  const chunkBounds = new Map<number, Bounds>();
  let maxW = 0;
  let maxH = 0;

  chunkIds.forEach(cid => {
    const feats = chunks.get(cid)!;
    const b = calculateBounds(feats);
    if (b) {
      chunkBounds.set(cid, b);
      if (b.width > maxW) maxW = b.width;
      if (b.height > maxH) maxH = b.height;
    }
  });

  // 3. Determine Grid
  const cols = Math.ceil(Math.sqrt(nChunks));
  const cellSizeX = maxW + gap;
  const cellSizeY = maxH + gap;

  // 4. Transform features
  const newFeatures: FeatureLike[] = [];
  
  chunkIds.forEach((cid, index) => {
    const row = Math.floor(index / cols);
    const col = index % cols;
    
    const targetX = col * cellSizeX;
    const targetY = -row * cellSizeY; // Go down

    const bounds = chunkBounds.get(cid);
    if (!bounds) return;

    const feats = chunks.get(cid)!;
    
    // Shift: x_new = x - minX + targetX
    // This places the min-point of the chunk at targetX/Y relative to origin (0,0) of that cell
    const offsetX = targetX - bounds.minX;
    const offsetY = targetY - bounds.minY; 

    feats.forEach(f => {
        const geometry = f.geometry;
        let newGeometry = geometry;

        if (geometry?.type === 'Polygon') {
          const nextCoordinates = geometry.coordinates.map((ring) =>
            ring.map((pt): Position => [pt[0] + offsetX, pt[1] + offsetY, ...pt.slice(2)])
          );
          newGeometry = { ...geometry, coordinates: nextCoordinates };
        } else if (geometry?.type === 'MultiPolygon') {
          const nextCoordinates = geometry.coordinates.map((poly) =>
            poly.map((ring) =>
              ring.map((pt): Position => [pt[0] + offsetX, pt[1] + offsetY, ...pt.slice(2)])
            )
          );
          newGeometry = { ...geometry, coordinates: nextCoordinates };
        }

        newFeatures.push({
            ...f,
            geometry: newGeometry,
            // Keep original properties
        });
    });
  });

  return {
      features: newFeatures,
      bounds: calculateBounds(newFeatures)
  };
}

export interface GraphNode {
  id: number;
  position: [number, number];
  color: [number, number, number];
  properties: Record<string, unknown>;
}

export interface GraphEdge {
  sourcePosition: [number, number];
  targetPosition: [number, number];
  color: [number, number, number];
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

function getPolygonCentroid(geometry: Geometry | null | undefined): [number, number] | null {
  if (!geometry) return null;

  let points: Position[] = [];

  if (geometry.type === 'Polygon') {
    points = geometry.coordinates[0] ?? [];
  } else if (geometry.type === 'MultiPolygon') {
    points = geometry.coordinates[0]?.[0] ?? [];
  } else {
    return null;
  }

  if (!points || points.length < 3) return null; // Need at least a triangle

  let area = 0;
  let cx = 0;
  let cy = 0;
  
  // Shoelace formula for centroid
  // points usually has the last point == first point in GeoJSON
  // We iterate up to length - 1
  const len = points.length - 1;
  
  // If the polygon is degenerate (start != end or too few points), fallback
  if (len < 3) return [points[0][0], points[0][1]];

  for (let i = 0; i < len; i++) {
    const [x0, y0] = points[i];
    const [x1, y1] = points[i + 1];
    
    const crossProduct = (x0 * y1 - x1 * y0);
    area += crossProduct;
    cx += (x0 + x1) * crossProduct;
    cy += (y0 + y1) * crossProduct;
  }
  
  area /= 2;
  
  // If area is 0 (collinear points or empty), fallback to vertex average
  if (Math.abs(area) < 1e-9) {
      let sumX = 0, sumY = 0;
      for (let i = 0; i < len; i++) {
        sumX += points[i][0];
        sumY += points[i][1];
      }
      return [sumX / len, sumY / len];
  }

  cx /= (6 * area);
  cy /= (6 * area);

  return [cx, cy];
}

export function generateGraphData(features: FeatureLike[]): GraphData {
  const nodes: GraphNode[] = [];
  const edges: GraphEdge[] = [];
  // Map Key: "layout_id-node_id"
  const nodeMap = new Map<string, GraphNode>();

  // 1. Create Nodes
  features.forEach(f => {
    const props = f.properties;
    if (!props) return;
    const nodeIdRaw = props.node_id;
    const layoutIdRaw = props.layout_id;
    const nodeId = typeof nodeIdRaw === 'number' ? nodeIdRaw : Number(nodeIdRaw);
    const layoutId = typeof layoutIdRaw === 'number' ? layoutIdRaw : Number(layoutIdRaw);
    if (!Number.isFinite(nodeId) || !Number.isFinite(layoutId)) return;

    const centroid = getPolygonCentroid(f.geometry);
    if (!centroid) return;

    const chunkIdRaw = props.chunk_id ?? 0;
    const chunkId = typeof chunkIdRaw === 'number' ? chunkIdRaw : Number(chunkIdRaw);
    const color = getChunkColor(Number.isFinite(chunkId) ? chunkId : 0);

    const node: GraphNode = {
      id: nodeId,
      position: centroid,
      color: color,
      properties: props
    };

    nodes.push(node);
    const key = `${layoutId}-${nodeId}`;
    nodeMap.set(key, node);
  });

  // 2. Create Edges
  nodes.forEach(node => {
    const neighbors = node.properties.neighbors;
    const layoutIdRaw = node.properties.layout_id;
    const layoutId = typeof layoutIdRaw === 'number' ? layoutIdRaw : Number(layoutIdRaw);
    
    if (Array.isArray(neighbors)) {
      neighbors.forEach((nid) => {
        const neighborId = typeof nid === 'number' ? nid : Number(nid);
        if (!Number.isFinite(neighborId) || !Number.isFinite(layoutId)) return;
        // Only draw edge if neighbor ID > node ID (to avoid duplicates in undirected graph)
        // AND both are in the same layout
        if (neighborId > node.id) {
          const key = `${layoutId}-${neighborId}`;
          const neighborNode = nodeMap.get(key);
          
          if (neighborNode) {
            edges.push({
              sourcePosition: node.position,
              targetPosition: neighborNode.position,
              color: [128, 128, 128] // Grey edges
            });
          }
        }
      });
    }
  });

  return { nodes, edges };
}
