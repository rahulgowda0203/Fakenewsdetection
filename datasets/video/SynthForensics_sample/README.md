---
pretty_name: SynthForensics
language:
- en
license: cc-by-nc-4.0
task_categories:
- video-classification
tags:
- deepfake
- deepfake-detection
- synthetic-video
- forensics
- video
- face
- text-to-video
- image-to-video
size_categories:
- n<1K
viewer: true
---


<h1 align="center">SynthForensics: Benchmarking and Evaluating People-Centric Synthetic Video Deepfakes</h1>

<p align="center">
  <a href="LICENSE.md"><img src="https://img.shields.io/badge/Code%20License-MIT-yellow.svg" alt="Code License: MIT"></a>
  <a href="https://creativecommons.org/licenses/by-nc/4.0/"><img src="https://img.shields.io/badge/Dataset%20License-CC%20BY--NC%204.0-blue.svg" alt="Dataset License: CC BY-NC 4.0"></a>
  <a href="https://huggingface.co/datasets/SynthForensics/SynthForensics"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Datasets-SynthForensics-orange" alt="Hugging Face Datasets"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg" alt="Python 3.10+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg" alt="PyTorch 2.0+"></a>
</p>

<p align="center">
  <img src="https://huggingface.co/datasets/SynthForensics/SynthForensics/resolve/main/assets/50VIDS.gif" width="700">
</p>

<p align="center"><strong>Official Repository for the SynthForensics (<em>SF</em>) Benchmark</strong></p>

> **Note:** This is the **sample** release of SynthForensics, comprising **10 videos per generator** with their respective metadata in JSON format selected to broadly represent the diversity and characteristics of the full benchmark. It is intended for dataset preview, model selection, and preliminary evaluation purposes. The complete dataset is available at [huggingface.co/datasets/SynthForensics/SynthForensics](https://huggingface.co/datasets/SynthForensics/SynthForensics).

## Table of Contents

- [Abstract](#abstract)
- [Repository Overview](#-repository-overview)
- [Dataset Overview](#-dataset-overview)
  - [Construction Pipeline](#construction-pipeline)
  - [Key Highlights](#key-highlights)
  - [Generators](#generators)
  - [Statistics](#statistics)
  - [Download](#download)
  - [Dataset Structure](#dataset-structure)
  - [Dataset Splits](#dataset-splits)
  - [Dataset Metadata](#dataset-metadata)
  - [Version Descriptions](#version-descriptions)
- [Experimental Results](#-experimental-results)
  - [Comparative Analysis](#comparative-analysis)
  - [Detection](#detection)
- [License](#-license)

## Abstract

<p align="justify">Modern T2V/I2V generators synthesize people increasingly hard to distinguish from authentic footage, while current evaluation suites lag: legacy benchmarks target manipulation-based forgeries, and recent synthetic-video benchmarks prioritize scale over realistic human depiction. We introduce SynthForensics, a people-centric benchmark of 20,445 videos from 8 T2V and 7 I2V open-source generators, paired-source from FF++/DFD reals, two-stage human-validated, in four compression versions with full metadata. In our paired-comparison human study, raters prefer SynthForensics in 71–77% of head-to-head comparisons against each of nine existing synthetic-video benchmarks, while facial-quality metrics fall within the FF++/DFD baseline range. Across 15 detectors and three protocols, face-based methods drop 13–55 AUC points (mean 27) from FF++ to SynthForensics and a further 23 under aggressive compression; fine-tuning closes the gap at a backward cost on legacy benchmarks; re-training shows synthetic and manipulation features largely disjoint for most detectors. We release dataset, pipeline, and code.</p>

## 📦 Repository Overview

This repository hosts the full benchmark stack: dataset access, per-generator generation scripts, detector training and evaluation pipelines, comparative experiments (face quality, landmark stability), and the human-perception survey infrastructure.

```
SynthForensics/
├── assets/                  # Visual assets used in this README (pipeline figure, sample GIF/MP4, experimental result figures)
├── splits/                  # Dataset splits (train.json, val.json, test.json)
└── src/
    ├── generators/          # Per-generator generation scripts (8 generators) for T2V/I2V dataset construction
    ├── detectors/           # Detector implementations with training and evaluation pipelines
    ├── experiments/         # Comparative analyses against existing benchmarks (Face_quality, Landmark_stability)
    └── survey/              # Self-contained Docker stack (backend, frontend, nginx, postgres) for the human-perception study, plus the analysis pipeline used for the reported results
```

Each subfolder ships with a dedicated README documenting the steps to execute and exactly reproduce the corresponding experiments, generation runs, and reported results.

## 🎬 Dataset Overview

**SynthForensics** is a rigorously curated, people-centric benchmark for detecting **purely synthetic video deepfakes** generated by modern Text-to-Video (T2V) and Image-to-Video (I2V) models. Unlike traditional benchmarks that focus on manipulation-based deepfakes (face swaps, expression reenactment), SynthForensics addresses the emerging threat of content that is **born synthetic** rather than altered.

### Construction Pipeline

<p align="center">
  <img src="https://huggingface.co/datasets/SynthForensics/SynthForensics_sample/resolve/main/assets/pipeline.jpg" width="900">
</p>

1. **Source Data Selection**: 1,363 pristine videos from FaceForensics++ and DeepFake Detection
2. **Structured Caption Extraction**: VLM (VideoLLaMA 3) extracts 8-field captions per source video
3. **Caption Validation**: Two-stage screening combining manual human review and LLM-based (Llama) filtering across seven thematic categories; flagged captions are iteratively rewritten until they pass both checks
4. **I2V Reference Frame Selection**: Human annotators select a reference frame from each source video for I2V conditioning
5. **Per-Generator Prompt Adaptation**: Positive prompts derived from the structured caption; negative prompts iteratively curated for artifact suppression
6. **Video Synthesis**: Generation in T2V or I2V mode with each generator's tuned hyperparameters
7. **Manual Video Validation**: Five-annotator review of anatomical, temporal, rendering, semantic, and ethical criteria; rejected videos trigger iterative prompt/parameter refinement until acceptance
8. **Compression Variants**: Accepted videos re-encoded into Raw, Canonical (CRF=0), CRF23, CRF40

### Key Highlights

- **20,445 unique videos** (~27.2 hours, ~1.93M frames) from 8 T2V and 7 I2V open-source generators
- **T2V + I2V coverage**: 10,904 text-to-video and 9,541 image-to-video synthetic videos
- **Paired-source protocol**: Each synthetic video paired with a real FF++/DFD source video to control semantic variables
- **Rigorous validation**: Two-stage human-in-the-loop quality control for semantic fidelity and ethical compliance
- **Four compression variants** per video (Raw, Canonical, CRF23, CRF40) for robustness evaluation  81,780 total files
- **Complete reproducibility**: Full metadata including prompts, generation parameters, and environment details for every video
- **Generators**: Wan2.1 (14B), CogVideoX (5B), SkyReels-V2 (14B), Self-Forcing (1.3B, T2V only), MAGI-1 (24B, distilled), LTX-2.3 (22B), daVinci-MagiHuman (15B, distilled), Helios (14B, distilled)

### Generators

The benchmark uses 8 T2V and 7 I2V state-of-the-art open-source generators. Self-Forcing is used in T2V mode only; all other generators appear in both T2V and I2V splits.

**T2V + I2V generators (7):**
- **[Wan2.1](https://github.com/Wan-Video/Wan2.1/tree/325cfd41e64639cf26d76f2b60601384909934cb)** — 14B (commit `325cfd41e64639cf26d76f2b60601384909934cb`)
- **[CogVideoX](https://github.com/zai-org/CogVideo/tree/5ab1e2449ffc8887ffad3ca3b9efd22ad7e356f7)** — 5B (commit `5ab1e2449ffc8887ffad3ca3b9efd22ad7e356f7`)
- **[SkyReels-V2](https://github.com/SkyworkAI/SkyReels-V2/tree/0790f89b91981a37ce1fdb0a76e7f10ffc8b9903)** — 14B (commit `0790f89b91981a37ce1fdb0a76e7f10ffc8b9903`)
- **[MAGI-1](https://github.com/SandAI-org/MAGI-1/tree/e7eeccf126f9294e7e119a90d1a57b9979faad36)** — 24B, distilled (commit `e7eeccf126f9294e7e119a90d1a57b9979faad36`)
- **[LTX-2.3](https://github.com/Lightricks/LTX-2/tree/9e8a28e17ac4dd9e49695223d50753a1ebda36fe)** — 22B (commit `9e8a28e17ac4dd9e49695223d50753a1ebda36fe`)
- **[daVinci-MagiHuman](https://github.com/GAIR-NLP/daVinci-MagiHuman/tree/f9764ced3f33017e75795f80adb9dcfd3ca68d38)** — 15B, distilled (commit `f9764ced3f33017e75795f80adb9dcfd3ca68d38`)
- **[Helios](https://github.com/PKU-YuanGroup/Helios/tree/5cb7ce9f95ce7df336ef072b9c8105321d6db54a)** — 14B, distilled (commit `5cb7ce9f95ce7df336ef072b9c8105321d6db54a`)

**T2V only (1):**
- **[Self-Forcing](https://github.com/guandeh17/Self-Forcing/tree/eb36b56aca91528bf5f1dcf36395e5c7151071e2)** — 1.3B (commit `eb36b56aca91528bf5f1dcf36395e5c7151071e2`)

### Statistics

| Metric | Value |
|--------|-------|
| Unique videos (T2V) | 10,904 |
| Unique videos (I2V) | 9,541 |
| Total unique synthetic videos | 20,445 |
| Total video files (4 compression versions) | 81,780 |
| Total unique frames | 1,934,097 |
| Total unique duration | ~27.2 hours |
| Landscape / Portrait videos | 16,349 / 4,096 |
| Resolution range | 384×640 to 1088×1920 |
| Frame rate range (FPS) | 8 to 25 |
| Duration range | 4 to 6 s |

### Download

🤗 **[Full dataset](https://huggingface.co/datasets/SynthForensics/SynthForensics)** — all 20,445 videos in four compression versions, plus metadata, captions, reference frames, and splits  
🤗 **[Sample dataset](https://huggingface.co/datasets/SynthForensics/SynthForensics_sample)** — smaller subset for quick exploration

Distributed via a gated Hugging Face repository — accept the terms of use to enable download.

### Dataset Structure
```
SynthForensics/
├── T2V/
│   ├── videos/
│   │   ├── raw/
│   │   │   ├── cogvideox/           # <ID>_cogvideox_t2v.mp4
│   │   │   ├── daVinci-MagiHuman/
│   │   │   ├── helios/
│   │   │   ├── ltx2-3/
│   │   │   ├── magi-1/
│   │   │   ├── self-forcing/
│   │   │   ├── skyreels-v2/
│   │   │   └── wan2-1/
│   │   ├── canonical/               # same per-generator structure
│   │   ├── crf23/
│   │   └── crf40/
│   └── metadata/
│       ├── cogvideox/               # <ID>_cogvideox_t2v.json
│       ├── daVinci-MagiHuman/
│       └── …                        # one sub-folder per generator
├── I2V/
│   ├── videos/
│   │   ├── raw/
│   │   │   ├── cogvideox/           # <ID>_cogvideox_i2v.mp4
│   │   │   ├── daVinci-MagiHuman/
│   │   │   ├── helios/
│   │   │   ├── ltx2-3/
│   │   │   ├── magi-1/
│   │   │   ├── skyreels-v2/
│   │   │   └── wan2-1/
│   │   ├── canonical/               # same per-generator structure
│   │   ├── crf23/
│   │   └── crf40/
│   ├── i2v_frames/                  # <ID>.png — reference frames used as conditioning input
│   └── metadata/
│       ├── cogvideox/               # <ID>_cogvideox_i2v.json
│       └── …                        # one sub-folder per generator
├── captions/                        # <ID>.json — dense captions for FF++ and DFD source videos
├── train.json
├── test.json
├── val.json
└── README.md
```

### Dataset Splits 
 
The SynthForensics dataset is divided into three standard splits for training, validation, and testing. 
Each split is provided as a JSON file: 
- `splits/train.json`   
- `splits/val.json`   
- `splits/test.json` 

**Split Protocol**: The dataset follows the official FaceForensics++ split assignment. Each source video ID from FaceForensics++ determines the split membership for all its corresponding synthetic versions across all five generators and all four compression variants. This ensures that synthetic videos generated from the same real source remain in the same split, preventing data leakage and enabling fair evaluation of generalization capabilities across different T2V models. In test split all DFD dataset is added for a total of 363 videos for generator.

### Dataset Metadata

Each video in **SynthForensics** is accompanied by detailed metadata to ensure reproducibility and support research on generation artifacts.

#### Example Metadata Entry
```json
{
  "generator": "LTX-2.3",
  "model_version": "ltx-2.3-22b-dev",
  "pipeline": "ti2vid_two_stages",
  "mode": "i2v",
  "checkpoint": "ltx-2.3-22b-dev.safetensors",
  "distilled_lora": "ltx-2.3-22b-distilled-lora-384.safetensors",
  "distilled_lora_strength": 0.5,
  "spatial_upsampler": "ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
  "text_encoder": "gemma-3-12b",
  "prompt": "A man in traditional white attire sitting and speaking, likely in a news studio setting. [...]",
  "negative_prompt": "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, [...]",
  "seed": 1364115978,
  "width": 1536, "height": 1024, "num_frames": 121, "frame_rate": 24,
  "num_inference_steps": 30,
  "video_cfg_scale": 3.0, "video_stg_scale": 1.0, "video_stg_blocks": [28],
  "video_rescale_scale": 0.7, "video_skip_step": 0, "a2v_guidance_scale": 3.0,
  "audio_cfg_scale": 7.0, "audio_stg_scale": 1.0, "audio_stg_blocks": [28],
  "audio_rescale_scale": 0.7, "audio_skip_step": 0, "v2a_guidance_scale": 3.0,
  "quantization": null, "dtype": "bfloat16",
  "source_caption": "000.json", "source_image": "000.png",
  "image_conditioning_frame_idx": 0, "image_conditioning_strength": 1.0, "image_conditioning_crf": 0,
  "orientation": "landscape",
  "environment": {
    "repo_url": "https://github.com/Lightricks/LTX-2.git",
    "repo_commit": "9e8a28e17ac4dd9e49695223d50753a1ebda36fe",
    "model_id": "ltx-2.3-22b-dev",
    "model_url": "https://huggingface.co/Lightricks/LTX-2.3",
    "model_hf_commit": "5a9c1c680bc66c159f708143bf274739961ecd08",
    "model_sha256": {
      "ltx-2.3-22b-dev.safetensors": "7ab7225325bc403448ea84b6db2269811a880e5118cd2ee2b6282a93d585016f",
      "...": "..."
    },
    "text_encoder_id": "gemma-3-12b",
    "text_encoder_url": "https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized",
    "text_encoder_hf_commit": "68f7ee4fbd59087436ada77ed2d62f373fdd4482",
    "python_version": "3.12.12", "torch_version": "2.7.1+cu128",
    "flash_attn_version": "2.8.3", "cuda_version": "12.8",
    "gpu_name": "NVIDIA A100 80GB PCIe",
    "key_deps": {"ltx-core": "1.0.0", "ltx-pipelines": "1.0.0", "xformers": "0.0.31.post1"}
  },
  "generation_time_seconds": 264.0
}
```

### Version Descriptions

- **Raw**: Direct, unprocessed generator output
- **Canonical**: Lossless re-encoding (H.264 CRF=0, YUV420p, BT.709) with uniform parameters
- **CRF23**: Light compression simulating standard online platforms
- **CRF40**: Heavy compression simulating aggressive distribution scenarios

## 📊 Experimental Results

### Comparative Analysis

#### Landmark Stability

Landmark completeness $\mathrm{Comp}_\tau$ as a function of the FAN heatmap-peak threshold $\tau$, on the face-positive subset of each benchmark. SynthForensics tracks the FF++/DFD real-video curves throughout the $\tau$ range, while the nine existing synthetic-video benchmarks fall well below. See **Section 4.1** of the paper for the full protocol and per-benchmark numbers.

<p align="center">
  <img src="https://huggingface.co/datasets/SynthForensics/SynthForensics_sample/resolve/main/assets/landmark_completeness_curves.png" width="700">
</p>

The exact reproduction code used to produce these results is available in [`src/experiments/Landmark_stability/`](src/experiments/Landmark_stability/).

#### Face Quality

Per-benchmark mean TOPIQ-NR-Face score $Q$ on the face-positive subset of each benchmark. SynthForensics sits within the FF++/DFD real-baseline range and is the closest entity to FF++ in 1D Wasserstein distance. See **Section 4.2** of the paper for the metric definition and per-benchmark numbers.

<p align="center">
  <img src="https://huggingface.co/datasets/SynthForensics/SynthForensics_sample/resolve/main/assets/face_quality_bars.png" width="700">
</p>

The exact reproduction code used to produce these results is available in [`src/experiments/Face_quality/`](src/experiments/Face_quality/).

#### Human Study

Paired-comparison human study with 118 retained participants and 4,121 ratings per question. Against existing benchmarks, SynthForensics is preferred on overall quality (Q1, 71.5%) and realism (Q2, 76.5%), and is flagged as fake less than half as often (Q3, 37.5% vs 84.1%). See **Section 4.3** of the paper for the protocol and per-benchmark breakdowns.

<p align="center">
  <img src="https://huggingface.co/datasets/SynthForensics/SynthForensics_sample/resolve/main/assets/human_results_v1_battle.png" width="700">
</p>

The exact reproduction code used to produce these results is available in [`src/survey/Experiments/`](src/survey/Experiments/).

### Detection

We evaluate 15 face-based and synthetic-video detectors on SynthForensics under three protocols: zero-shot evaluation, fine-tuning, and training from scratch. See **Section 5** of the paper for the protocols and the full per-detector results.

The exact reproduction code, training and evaluation pipelines, and per-detector setup instructions are available in [`src/detectors/`](src/detectors/).


#### Zero-Shot Performance on SynthForensics

<table>
<thead>
  <tr>
    <th rowspan="2"><strong>Detector</strong></th>
    <th rowspan="2"><strong>Trained on</strong></th>
    <th colspan="3"><strong>Legacy Benchmark Sets</strong></th>
    <th colspan="3"><strong>Primary Evaluation Sets</strong></th>
    <th colspan="2"><strong>Performance Gap</strong></th>
  </tr>
  <tr>
    <th><strong>FF++</strong></th>
    <th><strong>DFD</strong></th>
    <th><strong>CDF</strong></th>
    <th><strong>SF-FF++</strong></th>
    <th><strong>SF-DFD</strong></th>
    <th><strong>SF-CDF</strong></th>
    <th><strong>vs. FF++</strong></th>
    <th><strong>Mean</strong></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>CFM*</td>
    <td>FF++</td>
    <td>99.56</td>
    <td>95.21</td>
    <td>89.65</td>
    <td>73.54</td>
    <td>72.41</td>
    <td>66.03</td>
    <td>−26.02</td>
    <td>−24.15</td>
  </tr>
  <tr>
    <td>RECCE*</td>
    <td>FF++, DFDC, CDF, WildDF</td>
    <td>99.15†</td>
    <td>89.10‡</td>
    <td>99.94</td>
    <td>79.50</td>
    <td>78.63</td>
    <td>72.92</td>
    <td>−19.65</td>
    <td>−19.05</td>
  </tr>
  <tr>
    <td>ProDet*</td>
    <td>FF++</td>
    <td>98.77†</td>
    <td>90.10‡</td>
    <td>84.48</td>
    <td>75.35</td>
    <td>67.74</td>
    <td>69.42</td>
    <td>−23.41</td>
    <td>−20.28</td>
  </tr>
  <tr>
    <td>UCF*</td>
    <td>FF++</td>
    <td>99.50†</td>
    <td>94.50</td>
    <td>82.40‡</td>
    <td>75.41</td>
    <td>74.32</td>
    <td>75.39</td>
    <td>−24.09</td>
    <td>−17.09</td>
  </tr>
  <tr>
    <td>Effort*</td>
    <td>FF++</td>
    <td>98.11†</td>
    <td>96.50</td>
    <td>95.60</td>
    <td>70.31</td>
    <td>64.53</td>
    <td>60.21</td>
    <td>−27.79</td>
    <td>−31.72</td>
  </tr>
  <tr>
    <td>LAA-Net*</td>
    <td>FF++</td>
    <td>99.96</td>
    <td>98.43</td>
    <td>95.40</td>
    <td>69.42</td>
    <td>59.07</td>
    <td>47.48</td>
    <td>−30.54</td>
    <td>−39.27</td>
  </tr>
  <tr>
    <td>GenD*</td>
    <td>FF++</td>
    <td>98.97†</td>
    <td>97.00</td>
    <td>96.00</td>
    <td>82.35</td>
    <td>82.27</td>
    <td>68.83</td>
    <td>−16.62</td>
    <td>−19.51</td>
  </tr>
  <tr>
    <td>AltFreezing</td>
    <td>FF++</td>
    <td>98.60</td>
    <td>98.50</td>
    <td>89.50</td>
    <td>56.12</td>
    <td>71.65</td>
    <td>50.12</td>
    <td>−42.48</td>
    <td>−36.23</td>
  </tr>
  <tr>
    <td>FTCN</td>
    <td>FF++</td>
    <td>99.70</td>
    <td>94.40‡</td>
    <td>86.90</td>
    <td>44.76</td>
    <td>59.82</td>
    <td>53.69</td>
    <td>−54.94</td>
    <td>−40.91</td>
  </tr>
  <tr>
    <td>GenConViT</td>
    <td>FF++, DFDC, TM, CDF, TIMIT</td>
    <td>99.60</td>
    <td>99.95†</td>
    <td>98.10</td>
    <td>83.61</td>
    <td>90.68</td>
    <td>65.74</td>
    <td>−15.99</td>
    <td>−19.21</td>
  </tr>
  <tr>
    <td>DFD-FCG</td>
    <td>FF++</td>
    <td>99.57</td>
    <td>92.94†</td>
    <td>95.00</td>
    <td>86.57</td>
    <td>82.63</td>
    <td>77.05</td>
    <td>−13.00</td>
    <td>−13.75</td>
  </tr>
  <tr>
    <td>FakeSTormer</td>
    <td>FF++</td>
    <td>99.90</td>
    <td>98.90</td>
    <td>96.50</td>
    <td>72.59</td>
    <td>72.83</td>
    <td>64.30</td>
    <td>−27.31</td>
    <td>−28.53</td>
  </tr>
  <tr>
    <td colspan="10"><em>AI-generated video detectors (no legacy benchmark results)</em></td>
  </tr>
  <tr>
    <td>MM-Det</td>
    <td>DVF</td>
    <td>—</td>
    <td>—</td>
    <td>—</td>
    <td>49.17</td>
    <td>55.16</td>
    <td>41.26</td>
    <td>—</td>
    <td>—</td>
  </tr>
  <tr>
    <td>NSG-VD</td>
    <td>K400, Pika</td>
    <td>—</td>
    <td>—</td>
    <td>—</td>
    <td>58.95</td>
    <td>52.13</td>
    <td>70.58</td>
    <td>—</td>
    <td>—</td>
  </tr>
  <tr>
    <td>D3</td>
    <td>—</td>
    <td>—</td>
    <td>—</td>
    <td>—</td>
    <td>48.85</td>
    <td>49.40</td>
    <td>61.80</td>
    <td>—</td>
    <td>—</td>
  </tr>
</tbody>
</table>

*Frame-level detector. †Our evaluation. ‡Third-party results.

#### Fine-Tuning Results

Fine-tuning efficacy: zero-shot vs. fine-tuned AUC on SF-FF++ (Gain = FT − ZS) and backward AUC on FF++.

<table>
<thead>
  <tr>
    <th><strong>Detector</strong></th>
    <th><strong>Zero-Shot<br/>(on SF-FF++)</strong></th>
    <th><strong>Fine-Tuned<br/>(on SF-FF++)</strong></th>
    <th><strong>Gain (Δ)</strong></th>
    <th><strong>Backward<br/>(on FF++)</strong></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>RECCE*</td>
    <td>79.50</td>
    <td>98.04</td>
    <td>+18.54</td>
    <td>88.83</td>
  </tr>
  <tr>
    <td>ProDet*</td>
    <td>75.35</td>
    <td>96.14</td>
    <td>+20.79</td>
    <td>84.15</td>
  </tr>
  <tr>
    <td>UCF*</td>
    <td>75.41</td>
    <td>96.77</td>
    <td>+21.36</td>
    <td>92.55</td>
  </tr>
  <tr>
    <td>Effort*</td>
    <td>70.31</td>
    <td>96.40</td>
    <td>+26.09</td>
    <td>92.36</td>
  </tr>
  <tr>
    <td>GenD*</td>
    <td>82.35</td>
    <td>80.60</td>
    <td>−1.75</td>
    <td>98.23</td>
  </tr>
  <tr>
    <td>AltFreezing</td>
    <td>56.12</td>
    <td>99.12</td>
    <td>+43.00</td>
    <td>65.52</td>
  </tr>
  <tr>
    <td>FTCN</td>
    <td>44.76</td>
    <td>98.36</td>
    <td>+53.60</td>
    <td>57.07</td>
  </tr>
  <tr>
    <td>GenConViT</td>
    <td>83.61</td>
    <td>97.56</td>
    <td>+13.95</td>
    <td>86.39</td>
  </tr>
  <tr>
    <td>DFD-FCG</td>
    <td>86.57</td>
    <td>92.81</td>
    <td>+6.24</td>
    <td>99.37</td>
  </tr>
</tbody>
</table>

*Frame-level detector.

#### Training from Scratch

In-domain consists in all videos generated from 4 generators (CogVideoX, Wan2.1, LTX-2.3 and Magi-1) in both T2V and I2V modality, out-of-domain consists in all videos generated from 4 generators (SkyReels-V2, Self-Forcing, Helios and daVinci-MagiHuman) in both T2V and I2V modality when possible, and legacy consist in the mean of full legacy datasets (FaceForensics++, Deepfake Detection Dataset and Celeb-DF V2) transfer AUC %.

<table>
<thead>
  <tr>
    <th><strong>Detector</strong></th>
    <th><strong>In-Domain</strong></th>
    <th><strong>Out-of-Domain</strong></th>
    <th><strong>Legacy</strong></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>RECCE*</td>
    <td>97.21</td>
    <td>94.65</td>
    <td>65.25</td>
  </tr>
  <tr>
    <td>ProDet*</td>
    <td>78.91</td>
    <td>80.92</td>
    <td>54.57</td>
  </tr>
  <tr>
    <td>UCF*</td>
    <td>96.84</td>
    <td>94.33</td>
    <td>60.82</td>
  </tr>
  <tr>
    <td>Effort*</td>
    <td>97.83</td>
    <td>94.64</td>
    <td>62.97</td>
  </tr>
  <tr>
    <td>GenD*</td>
    <td>56.81</td>
    <td>55.47</td>
    <td>50.11</td>
  </tr>
  <tr>
    <td>AltFreezing</td>
    <td>99.24</td>
    <td>98.78</td>
    <td>63.08</td>
  </tr>
  <tr>
    <td>FTCN</td>
    <td>98.14</td>
    <td>97.74</td>
    <td>53.75</td>
  </tr>
  <tr>
    <td>GenConViT</td>
    <td>94.36</td>
    <td>87.58</td>
    <td>60.19</td>
  </tr>
  <tr>
    <td>DFD-FCG</td>
    <td>95.52</td>
    <td>90.19</td>
    <td>83.41</td>
  </tr>
</tbody>
</table>

<!--

## 📖 Documentation

Comprehensive documentation is available in the [`docs/`](docs/) directory:

- [Dataset Format](docs/dataset_format.md)
- [Evaluation Protocols](docs/evaluation.md)
- [Training Guidelines](docs/training.md)
- [Metadata Schema](docs/metadata_schema.md)
- [Reproduction Guide](docs/reproduction.md)
---

-->

<!--
## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Reporting Issues

If you encounter any issues or have suggestions, please open an issue on GitHub with:
- Dataset version
- Reproduction steps
- Expected vs. actual behavior

---
-->

## ⚖️ License

The repository code is licensed under the [MIT License](LICENSE.md). The SynthForensics dataset is licensed under [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/); full terms are available on the [Hugging Face dataset page](https://huggingface.co/datasets/SynthForensics/SynthForensics).
