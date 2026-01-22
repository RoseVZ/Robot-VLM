# Vision-Language-Guided Motion Planning for Instruction-Based Navigation

A robotics navigation system that combines Vision-Language Models (VLMs) with classical path planning algorithms to enable autonomous object search in unexplored indoor environments.

##  Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Experimental Results](#experimental-results)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)

---

##  Overview

This project implements a **Progressive Confidence Navigation System** that enables a mobile robot to autonomously find specific objects (e.g., "find the fridge") in unexplored indoor apartments without prior maps or human intervention.

### Key Innovation

**Progressive Confidence Assessment**: Mimics human behavior by moving closer to verify uncertain object detections, combining:
- **GroundingDINO** for fine-grained object detection with spatial localization
- **GPT-4V** for semantic reasoning and room verification
- **A\*/Dijkstra** for efficient path planning
- **Frontier-based exploration** for systematic environment discovery

---

##  Features

### Multimodal Perception
-  **Open-vocabulary object detection** using GroundingDINO
-  **Semantic reasoning** with GPT-4V for room classification
-  **360° panoramic capture** (8 views at 45° intervals)
-  **Combined confidence scoring** (α=0.7 for detection, β=0.3 for semantics)

### Intelligent Navigation
-  **Two-stage verification framework**
  - Stage 1: Initial detection → navigate (≥0.80), approach (≥0.60), or explore (<0.60)
  - Stage 2: Closer verification with stricter thresholds (≥0.80)
-  **Real-time SLAM** with occupancy grid mapping (0.1m resolution)
- **Classical path planning** (A\*, Dijkstra's) with dynamic replanning
-  **Frontier-based exploration** using BFS wavefront detection

### Robust Design
- ✅ **Zero prior knowledge** required
- ✅ **False positive prevention** through progressive verification
- ✅ **Collision-aware navigation** with robot footprint (0.5m radius)
- ✅ **Fallback mechanisms** for API failures

---

##  System Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                      User Query                             │
│                  "Go to the fridge"                         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
         ┌───────────────────────────────┐
         │   Panoramic View Capture      │
         │   (8 views at 45° intervals)  │
         └───────────┬───────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
┌──────────────────┐    ┌─────────────────┐
│ GroundingDINO    │    │    GPT-4V       │
│ Object Detection │    │ Room Reasoning  │
│ (Spatial)        │    │ (Semantic)      │
└────────┬─────────┘    └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
         ┌───────────────────────┐
         │ Combined Confidence   │
         │ α·C_GD + β·C_VLM     │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │  Progressive Decision │
         │   Stage 1 → Stage 2   │
         └───────────┬───────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
┌──────────────┐          ┌──────────────┐
│  Navigate    │          │   Explore    │
│  (A* Path)   │          │  (Frontier)  │
└──────────────┘          └──────────────┘
```

---

##  Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, for faster inference)
- OpenAI API key or Anthropic API key

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/vlm-navigation.git
cd vlm-navigation
```

### Step 2: Install Dependencies
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

**requirements.txt:**
```
numpy>=1.21.0
pybullet>=3.2.0
pillow>=9.0.0
opencv-python>=4.5.0
openai>=1.0.0
anthropic>=0.7.0
python-dotenv>=0.19.0
matplotlib>=3.5.0
torch>=2.0.0
transformers>=4.30.0
```

### Step 3: Download Kenny 3D Models
```bash
# Download Kenny's 3D models
mkdir object_models
cd object_models

# Download from: https://kenney.nl/assets/furniture-kit
# Extract .obj files to this directory

cd ..
```

### Step 4: Configure API Keys

Create a `.env` file in the project root:
```bash
# .env
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
```

**Important:** Add `.env` to `.gitignore` to prevent accidentally committing API keys!

---



## Configuration

### Environment Layouts

Three layouts available:

1. **Single Room** (8m × 8m)
   - Simple kitchen with 5 objects
   - Baseline testing

2. **Multi-Room** (16m × 16m)
   - Living room, bedroom, kitchen
   - 12-15 objects
   - Room transition testing

3. **Cluttered** (20m × 20m)
   - 4 rooms with narrow passages
   - 20-25 objects
   - Stress testing


### Confidence Thresholds

Adjust in `vlm_room_verifier.py`:
```python
# Stage 1 thresholds
HIGH_CONFIDENCE = 0.80  # Navigate directly
MEDIUM_CONFIDENCE = 0.60  # Approach and verify

# Stage 2 threshold
VERIFICATION_THRESHOLD = 0.80  # After approaching

# Weighting
ALPHA = 0.7  # GroundingDINO weight
BETA = 0.3   # GPT-4V weight
```

---

## 📊 Experimental Results

### Detection Performance

| Method | Success Rate | False Positive Rate | Avg Confidence |
|--------|-------------|---------------------|----------------|
| **Combined (Ours)** | **92%** | **3%** | **0.85** |
| GroundingDINO only | 71% | 22% | 0.68 |
| GPT-4V only | 65% | 8% | 0.72 |
| Heuristic baseline | 45% | 35% | N/A |

### Path Planning Performance

**Single Room ("Go to fridge"):**

| Algorithm | Path Length | Planning Time | Success Rate |
|-----------|-------------|---------------|--------------|
| A* | 5.10m | 274.55ms | 100% |
| Dijkstra | 5.39m | 358.92ms | 100% |

**Multi-Room ("Go to table"):**

| Algorithm | Path Length | Planning Time | Success Rate |
|-----------|-------------|---------------|--------------|
| A* | 19.5m | 1032.88ms | 60% |
| Dijkstra | 21.3m | 1263.92ms | 50% |



---



##  Authors

- **Tanya Mehta** - [tm3517@columbia.edu](mailto:tm3517@columbia.edu)
- **Priyanka Rose Varghese** - [prv2108@columbia.edu](mailto:prv2108@columbia.edu)

---

## Acknowledgments

- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) for open-vocabulary detection
- [OpenAI GPT-4V](https://openai.com/research/gpt-4v-system-card) for vision-language reasoning
- [Kenny](https://kenney.nl/) for 3D furniture models
- [PyBullet](https://pybullet.org/) for physics simulation


