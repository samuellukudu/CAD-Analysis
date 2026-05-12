#!/usr/bin/env bash
# Bash required (not `sh`).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
API_PORT="${API_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-9001}"
UVICORN_HOST="${UVICORN_HOST:-0.0.0.0}"
UVICORN_WORKERS="${UVICORN_WORKERS:-1}"

_extra_env="${EXTRA_DEV_PORTS:-}"
DEV_PORTS=( "${API_PORT}" "${FRONTEND_PORT}" )
for _p in ${_extra_env}; do
    [[ -n "${_p}" ]] || continue
    DEV_PORTS+=( "${_p}" )
done

LOG_DIR="${RUN_DEV_LOG_DIR:-${REPO_ROOT}/logs}"
BACKEND_LOG="${LOG_DIR}/dev-backend.log"
FRONTEND_LOG="${LOG_DIR}/dev-frontend.log"
BACKEND_PID_FILE="${LOG_DIR}/dev-backend.pid"
FRONTEND_PID_FILE="${LOG_DIR}/dev-frontend.pid"

kill_port() {
    local port="$1"
    local pids
    pids="$(lsof -nP -iTCP:"${port}" -sTCP:LISTEN -t 2>/dev/null || true)"
    if [[ -z "${pids}" ]]; then
        pids="$(lsof -ti tcp:"${port}" 2>/dev/null || true)"
    fi
    if [[ -n "${pids}" ]]; then
        kill -9 ${pids} 2>/dev/null || true
    fi
}

free_dev_ports() {
    local round port
    for round in 1 2; do
        for port in "${DEV_PORTS[@]}"; do
            [[ -n "${port}" ]] || continue
            kill_port "${port}"
        done
        if [[ "${round}" -eq 1 ]]; then
            sleep 0.3
        fi
    done
}

init_session_log() {
    local path="$1"
    local title="$2"
    mkdir -p "$(dirname "${path}")"
    {
        printf '=== %s — %s — %s ===\n' "${title}" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$(hostname 2>/dev/null || echo unknown-host)"
    } > "${path}"
}

usage() {
    printf '%s\n' \
        "Usage: $0 [OPTION]..." \
        "" \
        "  (no args)          Dev: uvicorn --reload + npm run dev (Vue dev), background." \
        "  --production, -p   Prod: npm run build, uvicorn (no reload), static frontend from dist/." \
        "  -f, --foreground   Stay attached; logs to terminal and logs/*.log" \
        "  --stop             Stop saved PIDs and free dev ports." \
        "  -h, --help         This help." \
        "" \
        "Examples:  $0 --production    $0 -p" \
        "           $0 --production --foreground" \
        "" \
        "Requires bash. RUN_PROFILE=production same as --production." \
        "Env: API_PORT FRONTEND_PORT  UVICORN_HOST  UVICORN_WORKERS" \
        "     SKIP_NPM_INSTALL=1  SKIP_FRONTEND_BUILD=1  EXTRA_DEV_PORTS  RUN_DEV_LOG_DIR" \
        "" \
        "Prod UI URL: http://<host>:FRONTEND_PORT/ (from frontend/dist)" \
        "Set FASTAPI_API_URL=http://<host>:API_PORT in repo .env before build (frontend/vue.config.js)." \
        "Dev remote: open firewall for FRONTEND_PORT + API; set FASTAPI_API_URL to server IP."
}

stop_dev() {
    local pid f
    for f in "${BACKEND_PID_FILE}" "${FRONTEND_PID_FILE}"; do
        [[ -f "${f}" ]] || continue
        pid="$(tr -d ' \n\t' < "${f}" || true)"
        [[ -n "${pid}" ]] || continue
        if kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
        fi
        rm -f "${f}"
    done
    sleep 1
    free_dev_ports
    printf '%s\n' "run-dev: stopped (ports ${API_PORT}, ${FRONTEND_PORT} cleared)." >&2
}

MODE=daemon
PROFILE=dev
while [[ $# -gt 0 ]]; do
    case "$1" in
        --stop)
            stop_dev
            exit 0
            ;;
        -f | --foreground)
            MODE=foreground
            shift
            ;;
        --production | --prod | -p)
            PROFILE=production
            shift
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            printf 'run-dev: unknown option %q\n' "$1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ -n "${RUN_DEV_FOREGROUND:-}" && "${RUN_DEV_FOREGROUND}" != "0" ]]; then
    MODE=foreground
fi
if [[ "${RUN_PROFILE:-}" == "production" ]]; then
    PROFILE=production
fi

build_frontend_production() {
    if [[ "${SKIP_FRONTEND_BUILD:-0}" == "1" && -f "${REPO_ROOT}/frontend/dist/index.html" ]]; then
        printf '%s\n' "run-dev: SKIP_FRONTEND_BUILD=1 and dist/ exists — skipping npm run build." >&2
        return 0
    fi
    cd "${REPO_ROOT}/frontend"
    if [[ "${SKIP_NPM_INSTALL:-0}" != "1" ]]; then
        if [[ -f package-lock.json ]]; then
            npm ci
        else
            npm install
        fi
    fi
    NODE_ENV=production npm run build
}

free_dev_ports

mkdir -p "${LOG_DIR}"

if [[ "${PROFILE}" == "production" ]]; then
    printf '%s\n' "run-dev: production — building Vue (npm run build)…" >&2
    build_frontend_production
    init_session_log "${BACKEND_LOG}" "backend (uvicorn, production)"
    init_session_log "${FRONTEND_LOG}" "frontend (static dist, production)"
else
    init_session_log "${BACKEND_LOG}" "backend (uvicorn, dev)"
    init_session_log "${FRONTEND_LOG}" "frontend (npm run dev)"
fi

if [[ ! -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    if command -v uv >/dev/null 2>&1; then
        uv venv "${REPO_ROOT}/.venv" >/dev/null
    fi
fi

if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    UVICORN=( "${REPO_ROOT}/.venv/bin/python" -m uvicorn )
elif command -v uvicorn >/dev/null 2>&1; then
    UVICORN=( uvicorn )
else
    UVICORN=( python3 -m uvicorn )
fi

if [[ "${PROFILE}" == "production" ]]; then
    UVICORN_ARGS=( server.main:app --host "${UVICORN_HOST}" --port "${API_PORT}" )
    if [[ "${UVICORN_WORKERS}" =~ ^[0-9]+$ ]] && [[ "${UVICORN_WORKERS}" -gt 1 ]]; then
        UVICORN_ARGS+=( --workers "${UVICORN_WORKERS}" )
    fi
else
    UVICORN_ARGS=( server.main:app --reload --port "${API_PORT}" )
fi

if [[ "${MODE}" == "foreground" ]]; then
    printf 'run-dev: foreground — logs also in:\n  %s\n  %s\n' "${BACKEND_LOG}" "${FRONTEND_LOG}" >&2

    cleanup() {
        if [[ -n "${UVICORN_PID:-}" ]] && kill -0 "${UVICORN_PID}" 2>/dev/null; then
            kill "${UVICORN_PID}" 2>/dev/null || true
        fi
        free_dev_ports
    }
    trap cleanup EXIT INT TERM

    (
        cd "${REPO_ROOT}"
        "${UVICORN[@]}" "${UVICORN_ARGS[@]}" 2>&1 | tee -a "${BACKEND_LOG}"
    ) &
    UVICORN_PID=$!

    cd "${REPO_ROOT}/frontend"
    if [[ "${PROFILE}" == "production" ]]; then
        python3 -m http.server "${FRONTEND_PORT}" --directory "${REPO_ROOT}/frontend/dist" 2>&1 | tee -a "${FRONTEND_LOG}"
    else
        env FRONTEND_PORT="${FRONTEND_PORT}" npm run dev 2>&1 | tee -a "${FRONTEND_LOG}"
    fi
else
    (
        cd "${REPO_ROOT}" || exit 1
        nohup "${UVICORN[@]}" "${UVICORN_ARGS[@]}" >>"${BACKEND_LOG}" 2>&1 &
        echo $! >"${BACKEND_PID_FILE}"
    )
    (
        cd "${REPO_ROOT}/frontend" || exit 1
        if [[ "${PROFILE}" == "production" ]]; then
            nohup python3 -m http.server "${FRONTEND_PORT}" --directory "${REPO_ROOT}/frontend/dist" >>"${FRONTEND_LOG}" 2>&1 &
        else
            nohup env FRONTEND_PORT="${FRONTEND_PORT}" npm run dev >>"${FRONTEND_LOG}" 2>&1 &
        fi
        echo $! >"${FRONTEND_PID_FILE}"
    )

    if [[ "${PROFILE}" == "production" ]]; then
        printf '%s\n' \
            "run-dev: production stack started." \
            "  FastAPI:     http://${UVICORN_HOST}:${API_PORT}/" \
            "  Web UI:      http://<server-ip>:${FRONTEND_PORT}/" \
            "  Frontend:    static files from frontend/dist" \
            "  API base:    http://<server-ip>:${API_PORT}/api/…" \
            "  Logs:        ${BACKEND_LOG}" \
            "               ${FRONTEND_LOG}" \
            "  Stop:        ${REPO_ROOT}/run-dev.sh --stop" \
            "  Skip build:  SKIP_FRONTEND_BUILD=1 if dist/ already built" >&2
    else
        printf '%s\n' \
            "run-dev: dev stack started in the background (closing this terminal does not stop them)." \
            "  FastAPI (uvicorn):    http://127.0.0.1:${API_PORT}/" \
            "  Vue (webpack dev):    http://localhost:${FRONTEND_PORT}/" \
            "  FastAPI /api:         http://127.0.0.1:${API_PORT}/api/" \
            "  Production build:     ${REPO_ROOT}/run-dev.sh --production" \
            "  Logs:                 ${BACKEND_LOG}" \
            "                        ${FRONTEND_LOG}" \
            "  Stop:                 ${REPO_ROOT}/run-dev.sh --stop" \
            "  Live logs:            tail -f ${LOG_DIR}/dev-*.log" \
            "  Attached debug:       ${REPO_ROOT}/run-dev.sh --foreground" >&2
    fi
fi
