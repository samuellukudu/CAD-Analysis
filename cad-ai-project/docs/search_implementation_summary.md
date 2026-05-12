# Search Functionality Implementation Summary

This document outlines the end-to-end implementation of the CAD search system, covering backend logic, frontend UI, and the integration that links search results to drawing visualization.

## 1. Backend Implementation

The backend is built with **FastAPI** and provides endpoints for text and spatial searches, as well as file management.

### Search Logic (`server/routers/search.py` & `src/search_cad.py`)
- **Text Search Strategy**:
  - **Grep-First**: Uses system `grep` via `subprocess` to rapidly filter files containing the search term before parsing any JSON. This ensures high performance even with many large files.
  - **JSON Parsing**: Only files identified by grep are parsed to locate specific entities.
  - **Entity Support**: targeted extraction for `TEXT`, `MTEXT`, `ATTRIB`, and `ATTDEF` entities.
- **Spatial Search**:
  - Supports **Bounding Box** and **Point** queries.
  - Calculates entity bounding boxes for various types (`LINE`, `LWPOLYLINE`, `CIRCLE`, `INSERT`, `TEXT`).
- **Endpoints**:
  - `POST /search/text`: Accepts a query string and target paths. Returns matches with text, file path, and location `(x, y)`.
  - `POST /search/bbox`: Finds entities within a defined rectangle.
  - `POST /search/point`: Finds entities containing a specific point.
  - `GET /search/files`: Lists available JSON files for the search scope.

### File Management (`server/routers/files.py`)
- **Endpoints**:
  - `GET /api/stored-dxf-files`: Lists available DXF files in `storage/dxf`.
  - `GET /api/dxf-file/{filename}`: Serves the binary DXF content to the frontend viewer.
  - `POST /api/save-dxf-file`: Handles file uploads to the storage directory.

## 2. Frontend Implementation

The frontend uses **Vue.js 3** and **Quasar** to provide an interactive search experience.

### Search UI (`frontend/src/components/SearchBar.vue`)
- **Components**:
  - **File Selector**: Multi-select dropdown (`q-select`) to limit search scope.
  - **Search Input**: Text field triggering search on 'Enter' or button click.
  - **Results Dialog**: A modal (`q-dialog`) displaying results in a list.
- **Actions**:
  - **performSearch**: Sends an async request to the backend `text` endpoint.
  - **focusResult**: Triggered when a user clicks a result row. Emits a custom `focus-entity` event containing the target file and coordinates.

### Viewer & Coordination (`frontend/src/App.vue` & `ViewerPage.vue`)
- **Event Coordination (`App.vue`)**:
  - Listens for `@focus-entity` from the `SearchBar`.
  - **File Switching**: Determines if the target file is already open. If not, it opens the file in a new tab; otherwise, it switches to the existing tab.
  - **Pending Focus Logic**: If a file needs to load, the focus coordinates are stored in `pendingFocus`. The `onFileLoaded` handler checks this queue to apply the camera move once the viewer is ready.
- **Camera Control (`ViewerPage.vue`)**:
  - **`focusOnLocation(x, y)`**: Calls the underlying DXF Viewer API (`viewer.SetView`) to center the camera on the target coordinates with a predefined zoom level (width: 5000 units).

## 3. Linking Results to Drawings

The "Click-to-Focus" workflow connects the search list to the visual drawing:

1.  **User Click**: User selects a result (e.g., "Elevator Shaft" at `x: 100, y: 200` in `Building_A.dxf`).
2.  **Event Emission**: `SearchBar` emits `{ file: "Building_A.json", location: {x: 100, y: 200} }`.
3.  **Path Resolution**: `App.vue` converts the JSON path (used for search) to the corresponding DXF filename (`Building_A.dxf`).
4.  **State Management**:
    - If `Building_A.dxf` is active -> Immediate focus.
    - If `Building_A.dxf` is open but background -> Switch tab -> Immediate focus.
    - If `Building_A.dxf` is closed -> Open file -> Wait for load -> Focus.
5.  **Visualization**: The viewer camera animates or jumps to `(100, 200)`, bringing the search result into the center of the viewport.
