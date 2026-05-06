I have identified the cause of the "not useful" chunks. The current logic splits large clusters into smaller ones but fails to re-merge any tiny fragments that might be left over (e.g., a cluster of 60 nodes split into 50 + 10, where 10 might be too small).

### The Solution
I will modify `dxf2chunks.py` to add a **final cleanup pass** in the `create_chunks` function. Specifically:
1.  After all splitting operations (for any strategy: community, spatial, or kmeans), I will invoke `_merge_small_communities` one last time.
2.  This will force any leftover small fragments to be merged into their best-connected neighboring chunk, eliminating the tiny "noise" chunks shown in your images.

### Verification Plan
I will run the script on `data/house design.dxf` again and verify that:
1.  The number of chunks is reasonable (e.g., 3-5).
2.  There are no tiny chunks (with < 5 nodes) in the output.