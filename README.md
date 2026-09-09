---
title: PyRete ReAct Hardware Pricing Coordinator
emoji: ⚡
colorFrom: emerald
colorTo: slate
sdk: docker
app_port: 7860
pinned: false
---

# PyRete ReAct Hardware Pricing Coordinator

A **FARM stack** (FastAPI + React / Next.js) demonstration implementing the **ReAct (Reason + Act)** agent paradigm governed by the **PyRete** forward-chaining rule engine instead of LangChain's linear agent loop.

The application automatically resolves retail hardware purchase prices (MSRP) and calculates real-time currency conversions using web search and arithmetic tools, with session-isolated working memory and filesystem persistence.

---

## 🏗️ Architecture & Port Selection

```
┌────────────────────────────────────────────────────────┐
│               Docker Container (Port 7860)             │
│                                                        │
│  ┌────────────────────────┐  Rewrites (Internal)       │
│  │  Next.js 14 Frontend   │ ───────────────────────┐   │
│  │  Port 7860 (Public)    │                        │   │
│  └────────────────────────┘                        ▼   │
│                                           ┌──────────┐ │
│                                           │ FastAPI  │ │
│                                           │ Backend  │ │
│                                           │:8000     │ │
│  ┌─────────────────────────────────────┐  └──────────┘ │
│  │ wm_storage/ (Filesystem Persistence)│       ▲       │
│  │ 49 Pre-computed Product/Currency WM │───────┘       │
│  └─────────────────────────────────────┘               │
└────────────────────────────────────────────────────────┘
```

### Port Assignments:
* **`7860` (Frontend)**: Standard Hugging Face Spaces public port. Runs the production Next.js frontend and reverse-proxies `/api/*` and `/health` requests to the internal backend.
* **`8000` (Backend)**: Internal FastAPI service. Runs Uvicorn serving `/api/pricing`, `/health`, and Swagger interactive docs (`/docs`).

---

## 💾 Offline / Pre-Persisted Cache Mode

The application contains **49 pre-computed query evaluations** in [`wm_storage/`](file:///Users/mitchellkj19/freelance/PyRete/wm_storage) covering all 7 hardware products across all 7 supported international currencies:
* **Hardware Products**: Apple MacBook Pro 14", Dell XPS 15, Lenovo ThinkPad X1 Carbon, ASUS ROG Zephyrus G14, Apple iPad Pro M4, Samsung Galaxy Tab S9, HP Spectre x360.
* **Target Currencies**: EUR (€), GBP (£), JPY (¥), CAD (C$), AUD (A$), CHF (Fr), INR (₹).

### Forgiving Gemini Handling:
* **When `GEMINI_API_KEY` is present**: Live queries outside the pre-computed set can be executed through Gemini LLM and DuckDuckGo search.
* **When `GEMINI_API_KEY` is absent**: The engine operates smoothly in cached demo mode. All pre-computed combinations return instant answers and full working memory audit trails without throwing any fatal errors or halting the container.

---

## 🖥️ Local Docker Instructions (Mac Mini)

### Prerequisites:
1. Ensure **Docker Desktop for Mac** is installed and running on your Mac Mini.
   * Check status: Open Terminal and run `docker info`. If it prints system information, Docker is running.

### Method 1: Automated Script (Recommended)
Make the script executable and run:
```bash
chmod +x run_docker.sh
./run_docker.sh
```

### Method 2: Manual Commands
1. **Build the image**:
   ```bash
   docker build -t pyrete-farm .
   ```

2. **Run the container**:
   ```bash
   # Run in demo mode (using pre-persisted cache):
   docker run --rm -it -p 7860:7860 -p 8000:8000 --name pyrete-app pyrete-farm

   # OR run with a live Gemini API key:
   docker run --rm -it -p 7860:7860 -p 8000:8000 -e GEMINI_API_KEY="your-key-here" --name pyrete-app pyrete-farm
   ```

3. **Open in your browser**:
   * **Dashboard**: [http://localhost:7860](http://localhost:7860)
   * **FastAPI Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
   * **Backend Health**: [http://localhost:7860/health](http://localhost:7860/health)

4. **Stop the container**:
   * Press `Ctrl+C` in the terminal, or in another terminal run:
     ```bash
     docker stop pyrete-app
     ```

---

## 🤗 Hugging Face Spaces Deployment Instructions

Hugging Face Spaces supports custom Docker images on its free CPU tier.

### Step 1: Create a Space on Hugging Face
1. Navigate to [huggingface.co/new-space](https://huggingface.co/new-space).
2. Enter a **Space name** (e.g., `pyrete-hardware-pricing`).
3. Under **Space SDK**, select **Docker** (Blank template).
4. Select **Public** visibility.
5. Click **Create Space**.

### Step 2: Push Your Code to the Space
In your local terminal on your Mac Mini:

```bash
# Add Hugging Face Space git remote (replace <username> and <space-name> with yours)
git remote add space https://huggingface.co/spaces/<username>/<space-name>

# Ensure changes are committed
git add Dockerfile start.sh run_docker.sh .dockerignore README.md frontend/ backend/ wm_storage/
git commit -m "Configure Docker container for Hugging Face Spaces launch"

# Push to Hugging Face
git push space main
```

*(If prompted for credentials, use your Hugging Face username and an Access Token with write permissions from `https://huggingface.co/settings/tokens`)*.

### Step 3: Monitor Build & Launch
1. Open your Space page on Hugging Face.
2. Under the **Building** tab, watch the container build.
3. Once built, Hugging Face automatically routes traffic to port **7860**.
4. Your application will be live at `https://huggingface.co/spaces/<username>/<space-name>`!
