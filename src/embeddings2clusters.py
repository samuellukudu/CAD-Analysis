import os
# Fix Numba threading issues by disabling parallelism
os.environ["NUMBA_NUM_THREADS"] = "1"
from functools import lru_cache
import pandas as pd
import numpy as np
import umap
from sklearn.cluster import DBSCAN, AgglomerativeClustering, HDBSCAN
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

def cluster_embeddings(index_file, 
                       method='agglomerative', 
                       umap_n_neighbors=32, 
                       visualize=False,
                       # Alg specific params
                       dbscan_eps=0.5,
                       agg_distance_threshold=2.0): # Lower = Stricter
    """
    Methods: 'dbscan', 'hdbscan', 'agglomerative'
    """
    
    # --- 1. Load & Reduce Data ---
    if not os.path.exists(index_file): return {}
    mtime = os.path.getmtime(index_file)
    return _cluster_embeddings_cached(
        index_file=index_file,
        mtime=mtime,
        method=method,
        umap_n_neighbors=umap_n_neighbors,
        visualize=visualize,
        dbscan_eps=dbscan_eps,
        agg_distance_threshold=agg_distance_threshold,
    )

@lru_cache(maxsize=8)
def _cluster_embeddings_cached(
    index_file: str,
    mtime: float,
    method: str,
    umap_n_neighbors: int,
    visualize: bool,
    dbscan_eps: float,
    agg_distance_threshold: float,
):
    _ = mtime
    df = pd.read_parquet(index_file)
    required_cols = {"layout_id", "chunk_id", "embedding"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(
            f"Embedding file missing required columns {sorted(required_cols)}: {index_file}"
        )
    matrix = np.stack(df['embedding'].values)
    
    # Handle tiny datasets: Skip UMAP and cluster everything together if < 3 chunks
    if len(df) < 3:
        print(f"Dataset too small for clustering ({len(df)} < 3). Assigning all to Cluster 0.")
        # Create dummy columns for consistency
        df['cluster_label'] = 0
        df['umap_x'] = 0.0
        df['umap_y'] = 0.0
        n_clusters = 1
        embedding_2d = np.zeros((len(df), 2)) # Dummy
    else:
        print(f"Reducing {len(df)} embeddings with UMAP...")
    
        # Adjust n_neighbors for small datasets to avoid warnings
        n_neighbors = umap_n_neighbors
        if len(df) <= n_neighbors:
            n_neighbors = max(2, len(df) - 1)
            print(f"Dataset size ({len(df)}) <= n_neighbors ({umap_n_neighbors}). Adjusted n_neighbors to {n_neighbors}.")
    
        # Set n_jobs=1 to avoid Numba threading conflicts and suppress warnings
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.0, n_components=3, metric='cosine', random_state=42, n_jobs=1)
        # reducer = PCA(n_components=0.90)
        embedding_2d = reducer.fit_transform(matrix)
        
        # --- 2. Apply Clustering ---
        print(f"Clustering using {method.upper()}...")
        
        if method == 'dbscan':
            clusterer = DBSCAN(eps=dbscan_eps, min_samples=2)
            labels = clusterer.fit_predict(embedding_2d)
            
        elif method == 'hdbscan':
            # min_cluster_size: Smallest grouping to be considered a cluster
            # min_samples: How conservative you want to be (larger = more noise/unclustered)
            clusterer = HDBSCAN(min_cluster_size=3, min_samples=3) 
            labels = clusterer.fit_predict(embedding_2d)
            
        elif method == 'agglomerative':
            # distance_threshold is the linkage distance above which clusters will not be merged.
            # Since UMAP output is not normalized 0-1, you usually need to experiment.
            # Start around 1.0 - 5.0 for UMAP data.
            clusterer = AgglomerativeClustering(
                n_clusters=None, 
                distance_threshold=agg_distance_threshold,
                linkage='ward' # 'ward' minimizes variance (makes compact blobs)
            )
            labels = clusterer.fit_predict(embedding_2d)
            
        else:
            raise ValueError("Unknown method")
    
        # Save Results
        df['cluster_label'] = labels
        df['umap_x'] = embedding_2d[:, 0]
        df['umap_y'] = embedding_2d[:, 1]
        
        # Stats
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"Found {n_clusters} clusters.")

    # --- 3. Visualization ---
    if visualize:
        plt.figure(figsize=(10, 6))
        # Plot Noise (-1)
        noise = df[df['cluster_label'] == -1]
        plt.scatter(noise['umap_x'], noise['umap_y'], c='lightgrey', s=15, label='Noise')
        # Plot Clusters
        clustered = df[df['cluster_label'] != -1]
        if not clustered.empty:
            sc = plt.scatter(clustered['umap_x'], clustered['umap_y'], 
                             c=clustered['cluster_label'], cmap='tab20', s=15)
            plt.colorbar(sc, label='Cluster ID')
        plt.title(f"Method: {method.upper()} | Clusters: {n_clusters}")
        plt.show()

    # --- 4. Format Output (Your Standard Format) ---
    formatted_groups = {}
    for cluster_id, group_df in df.groupby('cluster_label'):
        if cluster_id == -1: continue # Skip noise
        
        layout_map = {}
        for _, row in group_df.iterrows():
            l_id = int(row['layout_id'])
            c_id = int(row['chunk_id'])
            if l_id not in layout_map: layout_map[l_id] = {'chunk_ids': [], 'scores': []}
            layout_map[l_id]['chunk_ids'].append(c_id)
            layout_map[l_id]['scores'].append(1.0) # Dummy score for clustering
            
        formatted_groups[int(cluster_id)] = layout_map
        
    # --- Add Summary Statistics to Output ---
    summary = {
        "method": method,
        "total_clusters": n_clusters,
        "total_chunks": len(df),
        "cluster_sizes": {cid: len(group) for cid, group in df.groupby('cluster_label') if cid != -1},
        "noise_count": len(df[df['cluster_label'] == -1])
    }
    
    return {"clusters": formatted_groups, "summary": summary}

# --- USAGE EXAMPLES ---

# OPTION A: HDBSCAN (Recommended)
# Best for finding groups without worrying about parameters
# groups_hdb = cluster_embeddings(INDEX_FILE, method='hdbscan', agg_distance_threshold=1.5)

# OPTION B: Agglomerative
# Best if you want to enforce a strict "tightness"
# groups_agg = cluster_embeddings(INDEX_FILE, method='agglomerative', agg_distance_threshold=2.0)
