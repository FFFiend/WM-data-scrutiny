# WM Training Data Coverage Audit

Part of [Physical Reasoning](https://physicalreasoning.ai), an open source research collective studying world models.

Systematic coverage analysis of open-source manipulation datasets against a fixed taxonomy of physical phenomena relevant to world model training. We tag 1,300 video clips from 5 datasets across 2 data domains (real robot teleoperation, simulation) and measure per-phenomenon coverage rates. The goal is to identify what physical phenomena are well-represented in existing robot data, where the gaps are, and whether simulation data can fill them.

## Key Findings

| Category | Robot (n=1000) | Sim (n=300) |
|---|---|---|
| Contact events | 99.2% | 99.7% |
| Object type | 98.6% | 99.7% |
| Multi-object interaction | **7.8%** | **36.7%** |
| Embodiment | 100% | 100% |
| Dynamics | 100% | 99.7% |
| Occlusion | 0% | 0% |
| Failure / recovery | 0.1% | 0% |
| Scene complexity | 98.5% | 99.7% |

- **Multi-object interaction** (stacking, insertion, collision) is the primary gap in robot data. Simulation data (RoboCasa) provides partial coverage at 36.7%, making it a candidate for sim-to-real augmentation.
- **Occlusion** registers 0% across all sources. This is partly a tagger limitation (the VLM operates on 8 uniformly-spaced frames and does not reason temporally) and partly a data gap (demonstrations are filmed from clear angles).
- **Failure and recovery** is 0% because existing datasets discard unsuccessful demonstrations. No existing source can supplement this; it requires intentional failure injection or modified collection protocols.

## Data Sources

| Domain | Source | Hub ID | Clips |
|---|---|---|---|
| Robot | DROID-100 | `lerobot/droid_100` | 100 |
| Robot | Bridge V2 | `IPEC-COMMUNITY/bridge_orig_lerobot` | 300 |
| Robot | Fractal / RT-1 | `IPEC-COMMUNITY/fractal20220817_data_lerobot` | 300 |
| Robot | Kuka | `IPEC-COMMUNITY/kuka_lerobot` | 300 |
| Sim | RoboCasa | `PEARLS-Lab/robocasa-composite-raw-videos` | 300 |

All robot datasets are accessed via their LeRobot/IPEC HuggingFace mirrors. RoboCasa videos are downloaded directly from HuggingFace.

## Taxonomy

Eight categories, each with subcategories:

| Category | Subcategories |
|---|---|
| Contact events | grasp, release, push, tool contact |
| Object type | rigid, articulated, deformable, fluid/granular |
| Multi-object interaction | stacking, insertion, collision |
| Embodiment | robot type, gripper vs hand, camera placement |
| Dynamics | high motion, low motion, idle |
| Occlusion | partial, full, re-entry |
| Failure / recovery | drop, slip, regrasp, retry |
| Scene complexity | single object, few objects, cluttered |

Tagging is multi-label: a single clip can be tagged with multiple subcategories across multiple categories.

## Method

Tagging proceeds in three stages:

1. **Cheap signals** (no GPU): Regex keyword matching on language instructions, gripper open/close detection from proprioception, frame differencing for motion/idle classification, and static embodiment metadata from source config.

2. **VLM tagging** (GPU, Modal): Qwen2.5-VL-7B-Instruct receives 8 uniformly-spaced frames per clip with a structured prompt and returns JSON-formatted labels against the taxonomy. Runs on A100 via [Modal](https://modal.com).

3. **Tag merging**: Tags from all layers are merged per clip. If any layer flags a subcategory, it is counted as present. Coverage = fraction of clips where at least one subcategory is tagged within a category.

## Repository Structure

```
.
├── config.py                  # Central configuration: sources, paths, parameters
├── taxonomy.py                # Taxonomy definition and ClipTags dataclass
├── sampling.py                # Clip sampling, frame extraction, video download
├── modal_app.py               # Modal GPU services (QwenVLM, GroundingDINO, DINOv2)
├── modal_client.py            # Local client wrappers for Modal services
├── signals/
│   └── cheap_signals.py       # Keyword matching, gripper events, frame differencing
├── tagging/
│   ├── vlm_tagger.py          # VLM prompt construction and response parsing
│   ├── detectors.py           # Grounding DINO and hand-object contact detection
│   └── validation.py          # Hand-label validation set selection and metrics
├── analysis/
│   ├── coverage_matrix.py     # Coverage matrix construction and gap analysis
│   ├── embeddings.py          # DINOv2 domain gap analysis
│   └── plotting.py            # Heatmaps, bar charts, gap analysis plots
├── scripts/
│   ├── run_sampling.py        # Step 1: sample clips, extract cheap signals
│   ├── run_tagging.py         # Step 2: VLM + detector tagging, merge, validation set
│   ├── run_analysis.py        # Step 3: coverage matrix, plots, findings
│   └── bulk_download.py       # Utility: parallel video download for IPEC datasets
├── results/
│   ├── coverage_matrix_source.csv
│   ├── coverage_matrix_group.csv
│   ├── gap_analysis.csv
│   ├── subcategory_coverage.csv
│   ├── findings.txt
│   ├── plots/                 # Generated figures (PNG + PDF)
│   └── audit/                 # Validation annotation sheet
└── requirements.txt
```

## Setup

### Prerequisites

- Python 3.11+
- A [Modal](https://modal.com) account (free tier works) for GPU inference
- HuggingFace account for dataset access

### Installation

```bash
pip install -r requirements.txt
```

### Deploy Modal services

The VLM tagger and embedding models run on Modal. Deploy once:

```bash
modal deploy modal_app.py
```

This creates three GPU-backed services:
- `QwenVLM` (A100): Qwen2.5-VL-7B-Instruct for structured tagging
- `GroundingDINODetector` (A10G): Open-vocab object detection
- `DINOv2Embed` (A10G): Frame embeddings for domain gap analysis

Verify deployment:

```bash
modal run modal_app.py  # smoke test with a blank image
```

## Running the Pipeline

### Step 1: Download videos (IPEC datasets)

For Bridge V2, Fractal, and Kuka, videos must be downloaded first:

```bash
python scripts/bulk_download.py --sources bridge_v2,fractal,kuka --workers 8
```

DROID-100 and RoboCasa videos are downloaded lazily during sampling.

### Step 2: Sample clips and extract cheap signals

```bash
python scripts/run_sampling.py --sources droid,bridge_v2,fractal,kuka,robocasa --max-clips 300
```

Outputs `data/tags/cheap_tags.parquet`.

### Step 3: Run VLM tagging

```bash
python scripts/run_tagging.py
```

Sends frames to Modal for VLM inference, merges all tag sources.
Outputs `data/tags/merged_tags.parquet` and a 50-clip validation annotation sheet.

Optional flags:
- `--skip-vlm`: Reuse existing VLM tags
- `--skip-detectors`: Skip Grounding DINO pass

### Step 4: Analyze coverage and generate plots

```bash
python scripts/run_analysis.py
```

Outputs coverage matrices, gap analysis, plots, and findings to `results/`.

Optional:
- `--skip-embeddings`: Skip DINOv2 domain gap analysis (faster)

## Extending

### Adding a new data source

1. Add an entry to `SOURCES` in `config.py` with the source name, group, hub ID, and loader type.
2. If the source requires a custom loader, add a `_load_<name>()` function in `sampling.py` and register it in `sample_all_sources()`.
3. Re-run the pipeline from Step 2.

### Adding a new taxonomy category

1. Add the category and its subcategories to `CATEGORIES` in `taxonomy.py`.
2. Add keyword patterns (if applicable) to `KEYWORD_PATTERNS` in `config.py`.
3. Update the VLM prompt in `tagging/vlm_tagger.py`.
4. Re-run from Step 2.

## Known Limitations

- **Egocentric data not included.** EPIC-KITCHENS-100 and Ego4D require academic access for video. Keyword-only tagging on narrations is insufficient for reliable coverage estimates. Deferred to follow-up.
- **Occlusion detection.** The VLM tagger does not reliably detect occlusion from 8 still frames. A temporal detector on dense frame sequences would provide better estimates.
- **Failure/recovery is structurally absent.** Existing datasets discard unsuccessful demonstrations. This gap cannot be addressed through data mixing; it requires new collection protocols.
- **Domain gap not yet measured.** DINOv2 embedding distances between robot and sim frames have not been computed. This is needed to validate sim data as a supplement for multi-object interaction.

## Citation

If you use this codebase or taxonomy, please cite:

```bibtex
@misc{wm-data-scrutiny-2026,
  title={WM Training Data Coverage Audit},
  author={Zahid, Owais},
  year={2026},
  url={https://github.com/physicalreasoning/WM-data-scrutiny}
}
```

## Related Projects

This work is part of [Physical Reasoning](https://physicalreasoning.ai), an open source research collective studying world models. Other projects:

| Project | Description |
|---|---|
| [isotherm](https://github.com/physicalreasoning/isotherm) | Calibrated decision model for Kalshi daily temperature ladders, evaluated walk-forward against market prices. |
| [forward](https://github.com/physicalreasoning/forward) | Sealed forward tests for world models: entries frozen and hashed before each round opens. |
| [causal-jepa](https://github.com/physicalreasoning/causal-jepa) | A negative result on joint-embedding predictive architectures for prediction-market strike ladders. |

Read more at [physicalreasoning.ai/blog](https://physicalreasoning.ai/blog/).

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
