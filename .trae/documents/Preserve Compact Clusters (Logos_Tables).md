I will further improve the chunking logic to preserve **dense, compact structures** (like logos, complex tables, or equipment) as single chunks, preventing them from being arbitrarily fragmented.

### The Improvement
I will modify `dxf2chunks.py` to add a **"Compactness Check"** during the chunk splitting phase.

1.  **Logic**: Before splitting a large cluster (which normally happens if it exceeds the size limit), the script will calculate its **spatial footprint** (bounding box diagonal).
2.  **Condition**: If a cluster is **spatially small** (e.g., < 20% of the total layout size) but contains **many nodes** (high density), it will be identified as a "Compact Feature" (e.g., a detailed logo).
3.  **Action**: These compact clusters will be **preserved intact** instead of being forced to split. This ensures that semantic objects like the "China Mobile" logo remain as a single, coherent region of interest.

### Verification Plan
I will run the script on `data/20250702-景观详图组_t3.dxf` and check the logs. I expect to see debug messages indicating that compact clusters were preserved (e.g., "Preserved compact cluster of size 300..."). This will confirm that high-density details are being handled correctly.