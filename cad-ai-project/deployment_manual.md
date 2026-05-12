# CAD AI Project Deployment Manual

## 1. Server Specifications

| Component | Specification Requirements | Notes |
| :--- | :--- | :--- |
| **Operating System (OS)** | Linux (Ubuntu 20.04/22.04 or Debian 11/12) or macOS | Linux is highly recommended for production |
| **Processor (CPU)** | Minimum 4 Cores | 8+ Cores recommended for concurrent CAD processing |
| **Memory (RAM)** | Minimum 16 GB | 32 GB recommended for large DXF processing and VLM interactions |
| **Storage** | Minimum 50 GB (SSD) | Used for logs, dependencies, and caching |

## 2. Step-by-Step Environment Setup

### 2.1. System Prerequisites
Ensure you have `curl`, `git`, and build tools installed:
```bash
sudo apt update
sudo apt install -y curl git build-essential fontconfig libgl1-mesa-glx
```

### 2.2. Font Installation (Linux)
The CAD visualization component (`src/visualize_cad.py`) relies on CJK fonts for rendering Chinese text correctly. 
```bash
# Create fonts directory
sudo mkdir -p /usr/share/fonts/opentype/noto

# Download Noto Sans CJK SC (Simplified Chinese)
sudo curl -L -o /usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf "https://github.com/notofonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf"

# Refresh font cache
sudo fc-cache -f -v
```

### 2.3. Rust & resvg Setup (SVG to PNG Conversion)
The system uses `resvg` for high-fidelity SVG to PNG rendering. Set up Rust and install `resvg` using Aliyun mirrors:
```bash
# 1. Set mirrors (temporary for this session — or add to ~/.bashrc later)
export RUSTUP_UPDATE_ROOT="https://mirrors.aliyun.com/rustup/rustup"
export RUSTUP_DIST_SERVER="https://mirrors.aliyun.com/rustup"

# 2. Download and run the installer using the mirror
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# 3. Reload environment
source "$HOME/.cargo/env"

# 4. Update Rust
rustup update

# 5. Install resvg CLI
cargo install resvg
```

### 2.4. Python Environment (Backend)
Ensure Python 3.10+ is installed.
```bash
# Navigate to project root
cd /path/to/cad-ai-project

# Initialize virtual environment using uv or venv
python3 -m venv .venv
source .venv/bin/activate

# Install core requirements
pip install -r requirements.txt
```

### 2.5. Node.js & Vue Environment (Frontend)
Ensure Node.js 18+ is installed.
```bash
cd frontend

# Install dependencies
npm install

# Build frontend for production
npm run build
cd ..
```

## 3. Configuration Guides

Create a `.env` file in the project root based on the following configurations:

| Category | Environment Variable | Example Value | Description |
| :--- | :--- | :--- | :--- |
| **Server Ports** | `API_PORT` | `8000` | Backend API port |
| **Server Ports** | `FRONTEND_PORT` | `9001` | Frontend UI port |
| **Server Ports** | `UVICORN_HOST` | `0.0.0.0` | Host binding for Uvicorn |
| **Server Ports** | `UVICORN_WORKERS` | `4` | Number of Uvicorn workers |
| **Model Configs** | `VISION_MODEL` | `qwen-vl-max` | Preferred vision model |
| **Model Configs** | `DASHSCOPE_API_KEY` | `your_dashscope_api_key_here` | DashScope API key |
| **Model Configs** | `BASE_URL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | OpenAI-compatible base URL |
| **Optimization** | `TOOL_OUTPUT_SUMMARY_THRESHOLD` | `2500` | Threshold for summarizing tool outputs |

*Note: For the frontend to communicate with the backend, ensure `FASTAPI_API_URL=http://<server-ip>:8000` is accessible, or rely on the proxy setup.*

## 4. Service Start / Stop Commands

We utilize the provided `run-dev.sh` script for deployment.

**Start Service (Production Mode):**
```bash
# Runs uvicorn in the background and serves frontend static files via http.server
./run-dev.sh --production
```
*(Optionally use `-f` to run in the foreground: `./run-dev.sh --production -f`)*

**Stop Service:**
```bash
# Gracefully kills PIDs tracked in the logs/ directory and clears ports
./run-dev.sh --stop
```

## 5. Health Check Validation Procedures

1. **Backend API Health**: Navigate to `http://<server-ip>:8000/docs`. You should see the FastAPI Swagger UI loaded successfully.
2. **Frontend UI Health**: Navigate to `http://<server-ip>:9001/`. The Vue application should render without console errors (press F12 to check Developer Tools).
3. **Port Binding**: Run `lsof -i :8000` and `lsof -i :9001` to confirm services are actively listening.

## 6. Full List of 3rd Party Libraries

| Layer | Library / Package | Version / Description |
| :--- | :--- | :--- |
| **Backend (Python)** | `fastapi[standard]`, `uvicorn[standard]` | Web framework and ASGI server |
| **Backend (Python)** | `pydantic` | Data validation |
| **Backend (Python)** | `python-multipart` | Form data parsing |
| **Backend (Python)** | `ezdxf`, `PyMuPDF` | CAD DXF and PDF parsing |
| **Backend (Python)** | `opencv-python`, `scikit-image`, `pillow` | Image processing |
| **Backend (Python)** | `scikit-learn`, `scipy`, `umap-learn` | Scientific computing and ML |
| **Backend (Python)** | `pandas`, `openpyxl`, `pyarrow`, `fastparquet` | Data manipulation and I/O |
| **Backend (Python)** | `matplotlib` | Plotting and visualization |
| **Backend (Python)** | `celery`, `redis`, `flower` | Asynchronous task queue |
| **Backend (Python)** | `openai`, `dspy` | LLM integration and reasoning frameworks |
| **Backend (Python)** | `PyYAML`, `python-dotenv`, `requests` | Config and HTTP clients |
| **Backend (Python)** | `tqdm`, `termcolor` | CLI utilities |
| **Frontend (Vue.js)** | `@quasar/extras` | `^1.17.0` |
| **Frontend (Vue.js)** | `axios` | `^1.13.6` |
| **Frontend (Vue.js)** | `d3` | `^7.9.0` |
| **Frontend (Vue.js)** | `dxf-viewer` | `^1.0.42` |
| **Frontend (Vue.js)** | `markdown-it` | `^14.1.1` |
| **Frontend (Vue.js)** | `quasar` | `^2.18.6` |
| **Frontend (Vue.js)** | `three` | `^0.161.0` |
| **Frontend (Vue.js)** | `vue` | `^3.5.26` |

## 7. AI Models Used

1. **Vision Language Models (VLMs)**: The system utilizes OpenAI-compatible VLM endpoints (configured via `VISION_MODEL` and `BASE_URL` in `.env`). This is predominantly configured for DashScope's Qwen family (e.g., `qwen-vl-max` or `qwen-vl-plus`) for analyzing layout crops, resolving paddings (`padding_vlm.py`), and compliance checking (`image_analysis.py`).

## 8. Technical Logic Description

### De-identification (Data Minimization)
To comply with token constraints and remove unnecessary CAD bloat (which acts as a form of structural anonymization), the system uses a `minify_for_llm` function (in `src/extract_roi_data.py`). 
- **Coordinate Rounding**: Floats are aggressively rounded.
- **Visual Property Stripping**: Removes properties like `color`, `lineweight`, `thickness`, and `xdata` (which often contains custom CAD plugin user data or proprietary metadata).
- **Key Compression**: Re-keys large JSON objects to single characters (e.g., `ownerHandle` -> `pid`, `layer` -> `l`, `text` -> `txt`). This minimizes the payload passed to external LLMs.

### Localization & Bounding Box Logic
Localization relies on a multi-hop reasoning architecture that classifies the target intent and routes it to the appropriate extraction and refinement pipeline:
1. **Plan & Layout Extraction**: For large-scale general drawings (e.g., floor plans, elevations, sections, profiles), the system uses `src/occupancy_grids.py`. It builds an occupancy grid mapping structural elements to isolate the entire plan.
2. **Specific Region Extraction**: For localized sub-regions (e.g., specific rooms, details), the system uses `src/vector.py` to perform text seeding and vector matching to find the precise localized area.
3. **Progressive Refinement (VLM)**: `src/refine_bbox_vlm.py` manages the refinement logic based on the extraction type:
   - **Padding**: Applied to occupancy grid outputs (general plans) via `src/padding_vlm.py` to ensure cut-off edges are expanded and fully captured.
   - **Cropping**: Applied to vector outputs (sub-regions) to trim excess whitespace and isolate the specific target area accurately.
