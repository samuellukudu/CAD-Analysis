I will implement a **maximum chunk limit per layout** to resolve the issue of excessive fragmentation (e.g., > 100 chunks).

### The Solution
I will modify `dxf2chunks.py` to:
1.  **Add a `max_chunks` parameter**: I will introduce a hard cap (default **50**) on the number of chunks generated for any single layout.
2.  **Adaptive Clustering**: Inside `create_chunks`, I will adjust the target number of clusters (`target_k`) so it never exceeds `max_chunks`.
    - If a layout is massive and would normally generate 200 chunks, the script will now force it into 50 larger chunks.
    - The internal parameters (min/max community size) will automatically scale up to accommodate these larger chunks.
3.  **CLI Support**: I will add a `--max_chunks` argument to the command line so you can tune this limit easily.

### Verification Plan
After applying the changes, I will run the script on `data/20250702-景观详图组_t3.dxf` (the file you mentioned) and verify that:
1.  No layout produces more than 50 chunks.
2.  The output logs explicitly confirm the chunk count (e.g., "gAAG Chunks (50 Chunks)").