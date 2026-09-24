# Legacy modules — OS / CUDA / torch compatibility

Compatibility of the ML modules in `portable/src/{sound_processing,visual_processing,visual_generation}`
as candidates to port into `.wmplugin` plugins, and the input for a **universal
per-plugin installer** that picks the right `requirements` for the detected OS +
CUDA. Everything here was read from the actual code (file:line anchors given);
where a version is inferred rather than pinned it is marked.

## How the legacy app pins things today (baseline, from `portable/pyproject.toml`)

| Target | torch | CUDA | ONNX | GPU-only libs | extra ML |
|---|---|---|---|---|---|
| **macOS** | `2.2.2` (CPU) | none | `onnxruntime` (CPU) | — | *no diffusers/xformers/bnb; `visual_generation` excluded* |
| **Windows** | `2.2.2+cu118` | **11.8** | `onnxruntime-gpu` | `xformers 0.0.25.post1+cu118` | diffusers 0.33.1, transformers 4.56.1, bitsandbytes 0.45.5 |
| **Linux** | `2.4.0+cu12` | **12.x** (cudnn9-cuda-12) | `onnxruntime-gpu` | `xformers 0.0.27post2` | diffusers 0.33.1, transformers 4.56.1, bitsandbytes 0.45.5 |
| CPU profile | unpinned CPU | none | `onnxruntime` | — | — |

Cross-cutting facts that make most modules portable across the torch pins:
- Device is chosen **CUDA-or-CPU, never MPS** (`backend/general_utils.py:29-42`); every "mps" branch found in the trees is upstream HF code that the app never reaches.
- `torch.load(..., weights_only=True)` is used almost everywhere → safe against the torch 2.6 default change; works on both 2.2.2 and 2.4.0.
- No `flash-attn`/`triton`/`bitsandbytes`/custom `.cu` kernels in `sound_processing` or `visual_processing`. The only CUDA-only libs in play are **xformers** (SD/LatentSync), **bitsandbytes** (Hunyuan quant) and **flash-attn** (video backends).
- `visual_generation` is video-diffusion heavy and is **excluded from the macOS build** on purpose.

---

## Final selection (owner, Jul 2026)

The curated set to port. Everything not listed as a plugin/helper/extension is
**removed**.

**Standalone plugins**
- Sound (all kept — CPU-fast, all OS incl. mac): **CloneVoice**, **Separator**, **Analyser/DigitalSignature** (deepfake/watermark; audio + the video Analyser can be one "Analyser" plugin over both).
- Visual (win·linux, GPU): **LivePortrait**, **FaceSwap**, **Enhancement** (GFPGAN + RealESRGAN + AnimeSGAN in ONE plugin with a model selector), **Wav2Lip**, **LatentSync** (separate plugin from Wav2Lip), **Highlights**.

**Helpers — bundled inside the plugins above, NOT standalone plugins**
- **FaceRecognition** — face tracking/alignment shared by LipSync (Wav2Lip/LatentSync), FaceSwap, LivePortrait.
- **SyncNet** — active-speaker / audio-visual-sync scoring (AV offset + confidence, `sync_net/instance.py:96-114`), used by **Highlights** via `FaceRecognitionSpeaker` (`recognition.py:353-364`, called at `visual_generation/inference.py:2304`) to pick the talking face. Not invoked directly.
- **SceneDetect** — shot splitting, used by Highlights and others.

**Extensions over the built-in Object Detection (SAM2), not standalone**
- **Segmentation** and **RemoveBackground** — reuse the shipped SAM2; no model port.

**Removed for good**
- ImageGeneration (SD1.5+ControlNet), all VideoGeneration (Pyramid Flow, Wan2.1, HunyuanVideo I2V/T2V), Restyling, EbSynth, RemoveObject/ProPainter.
- **OmniGen** — leaning remove (too heavy, low value); drop unless the owner revisits.

---

## Decisions & porting strategy (owner, Jul 2026)

These override the raw audit below where they conflict.

- **Wrap the upstream, do not vendor the snapshot.** The `portable/` code is a
  frozen 2024 snapshot; porting it 1:1 re-imports its *upper* version caps. For
  each feature, build the plugin on the **maintained upstream** (pip/git, no
  upper pins) so the installer can put the newest torch that matches the user's
  CUDA. Use the old code only as a reference for the pipeline glue
  (pre/post-processing, cropping, muxing). Modern upstreams per feature:

  | Feature | Upstream to wrap (tracks latest torch/CUDA) |
  |---|---|
  | Audio separation | `python-audio-separator` (pip) or Demucs v4 |
  | Voice clone / TTS | OpenVoice v2 → or newer F5-TTS / XTTS (coqui-ai-TTS) |
  | Portrait animation | KwaiVGI/LivePortrait |
  | Face swap | insightface (version-neutral) or facefusion |
  | Upscale / restore | **spandrel** (GFPGAN/RealESRGAN on new torch) or Real-ESRGAN **ncnn-vulkan** (no torch, runs on mac via Metal); faces → CodeFormer |
  | Lip sync | LatentSync 1.5 / MuseTalk (new-CUDA) |
  | Image / video / identity | `diffusers` latest (SD/SDXL/Flux/ControlNet; video CogVideoX/Mochi/HunyuanVideo/Wan) + official `Wan-AI/Wan2.1`, `Tencent/HunyuanVideo`, `VectorSpaceLab/OmniGen2` |
  | VLM / highlights | `transformers` latest (SmolVLM2, Qwen2-VL) |
  | SAM | facebookresearch/sam2 — **already shipped** in the built-in Object Detection |

  The heavy diffusion repos (Wan/Hunyuan/Flux) have **no upper cap** — they
  *require* recent CUDA/torch; the cap only ever came from the snapshot.

- **macOS = audio only.** Waiting on CPU image/video processing is not
  acceptable in 2026, and MPS is disabled in this stack. So visual modules are
  **win·linux (GPU)**; macOS ships the audio plugins (fast on CPU) and, at most,
  ONNX-based features via onnxruntime **CoreML EP** (no torch) where the model
  actually runs there — CoreML is model-dependent (e.g. RetinaFace falls back to
  CPU). No torch model gets rewritten to CoreML/MLX. This also makes the
  eager-diffusers-import macOS blocker moot (visual isn't on mac).

- **Dropped, do not port:** Restyling (SD img2img) and EbSynth (obsolete; the
  EbSynth linux binary is a cu118 native artifact) — alongside the already-dropped
  RemoveObject (SAM2+ProPainter). Bucket **E (native binary) is removed.**

- **SAM features reuse the built-in Object Detection (SAM2), no model port:**
  - **RemoveBackground** = an *extension of the Object Detection plugin*: turn a
    SAM2 mask into an alpha/matte (no new weights; optional soft-edge matting
    later), not a standalone plugin.
  - **Segmentation / EAST** (text detection) = skip for now; if needed later,
    replace with a modern detector (PaddleOCR), don't port EAST.

---

## Per-module matrix

Legend — `gpu`: `no` (never) · `opt` (CPU works, GPU faster) · `req` (GPU mandatory).
`cuda`: `none` · `any` (11.8 or 12, or CPU) · `11.8+` (old-CUDA capable) · `12.x` (new-CUDA in practice).

### sound_processing — all light, torch-version-flexible, macOS-friendly
| Module | os | torch | cuda | gpu | Key deps / notes |
|---|---|---|---|---|---|
| CloneVoice / OpenVoice | linux·win·mac | 2.2.2–2.4.0 | any | opt | torch+librosa+gtts+faster-whisper. Latent fp16/CUDA line at `clone_voice/se_extractor.py:22` only via unused `vad=False`. |
| AudioAnalyser / DigitalSignature | linux·win·mac | any 2.x | none | no | tiny CPU net, bundled `signature.pkl`; analysis forces `cpu` (`inference.py:90`). |
| Separator / MDX | linux·win·mac | any 2.x (STFT only) | any | opt | **onnxruntime** model (`providers=None` → GPU if available else CPU, `model.py:452`); torch only for STFT. |

### visual_processing — mostly CPU-capable; a few GPU/CUDA-locked
| Module | os | torch | cuda | gpu | Key deps / notes |
|---|---|---|---|---|---|
| LivePortrait | linux·win·mac\* | 2.2.2–2.4.0 | any | opt | torch+onnx+insightface; `torch.compile` off by default; CPU→bf16 autocast. |
| FaceSwap (inswapper) | linux·win·mac\* | any | any | opt | pure onnxruntime+insightface; torch only probes CUDA. |
| Enhancement — GFPGAN | linux·win·mac\* | 2.2.2–2.4.0 | any | opt | basicsr `functional_tensor` breakage **already shimmed** (`gfpganer.py:9-21`) → works on both torchvision 0.17 & 0.19. |
| Enhancement — RealESRGAN / AnimeSGAN | linux·win | 2.2.2–2.4.0 | any | **req** | hard `half=True`; silently no-ops on CPU (`face_enhancer.py:60-64`). |
| LipSync — **Wav2Lip** | linux·win·mac\* | 2.2.2–2.4.0 | any | opt | itself needs no diffusers; enhancers (GPEN/GFPGAN) off on CPU. |
| LipSync — **LatentSync** | **linux** | **2.4.0** (diffusers 0.33.1 / transformers 4.56) | **12.x** | **req** | forced fp16 (`sync.py:56-88`), xformers guard `attention.py:261-263`. Only truly GPU-locked module here. |
| RemoveBackground (SAM) | — | — | — | — | **do not port** → build on the built-in Object Detection (SAM2): mask → alpha/matte, as an extension. |
| Segmentation (SAM + EAST) | — | — | — | — | SAM covered by Object Detection; **EAST skipped** (use PaddleOCR later if needed). |
| FaceRecognition | linux·win·mac\* | any | any | opt | insightface + onnx. |
| SyncNet | linux·win | 2.2.2–2.4.0 (<2.6) | **req** | **req** | **hardcoded `.cuda()`** (`sync_net/instance.py:39,78,…`) + bare `torch.load`. Not CPU/mac. |
| Analyser | linux·win·mac | n/a | none | no | onnxruntime CPU only. |
| ~~RemoveObject~~ (SAM2+ProPainter) | — | — | — | — | **dropped, do not port.** |

`*` Per the decision above, **visual modules are win·linux (GPU), not macOS** — so the `*` mac caveat is moot for shipping. Two audit notes still matter when porting: (1) each feature should be its **own plugin** so LatentSync's diffusers import can't sink Wav2Lip/FaceSwap (the eager import chain `inference.py:24` → `lip_sync/__init__.py:4` → `sync.py:4-5` otherwise couples them); (2) RealESRGAN/AnimeSGAN/SyncNet are GPU-`req`.

### visual_generation — heavy diffusion; the old-vs-new-CUDA fault line
| Module / backend | os | torch | cuda | gpu (VRAM) | Why it lands there |
|---|---|---|---|---|---|
| SceneDetect | linux·win·(mac) | n/a | none | no | pyscenedetect + opencv only. |
| ImageGeneration (SD1.5+ControlNet) | linux·win | 2.2.2+ | **11.8+** | opt→req (~6 GB) | SDPA+xformers+fp16; CPU fallback exists. |
| ~~Restyling~~ (SD img2img+GMFlow) | — | — | — | — | **dropped (obsolete), do not port.** |
| OmniGen (Phi-3) | linux·win (mac: maybe\*\*) | 2.2.2+ | **11.8+** | req (~6–12 GB) | SDPA Phi3, flash-attn **optional**; needs transformers 4.56 cache API. |
| Highlights (SmolVLM2+Whisper+Kosmos2) | linux·win | 2.2.2+ | **11.8+** | light (~2–6 GB) | bf16 pref, flash-attn optional; torchvision `grayscale_to_rgb` shim (`highlights/processing.py:27-31`). |
| **Pyramid Flow (SD3)** — video "Basic" | linux·win | 2.2.2+ | **11.8+** | req (~7 GB) | vendored, flash-attn **optional** (`use_flash_attn=False` default). The one video backend that runs without flash-attn → the Windows/legacy default. |
| **Wan2.1-T2V** | **linux** (win only w/ flash-attn) | **2.4** | **12.x** | req (~20 GB; 14B=80) | listed only if `FLASH_ATTN_2_AVAILABLE` (`inference.py:277`); bf16; CUDA-assert in attention. |
| **HunyuanVideo-I2V** | **linux** | **2.4** | **12.x** | req (80 GB / nf4<28) | **bitsandbytes 4bit/8bit** (CUDA-only) + flash-attn + diffusers-0.33.1-locked `HunyuanVideo*` classes. |
| **FastHunyuan-T2V** | **linux** | **2.4** | **12.x** | req (~20 GB, quant tiers) | same lock as Hunyuan-I2V. |
| ~~EbSynth~~ | — | — | — | — | **dropped (obsolete cu118 native binary), do not port.** |

`**` OmniGen is the only `visual_generation` backend with a live MPS code path (`generation/omnigen/pipeline.py:55`), but the app forces cuda/cpu, so macOS is only theoretical.

**The crux the user hit** — "new CUDA can't run the old modules; some models changed torch/cuda classes":
- **Old-CUDA (11.8) capable** (SDPA + xformers + fp16/bf16, stable since torch 2.0): ImageGeneration, OmniGen, Highlights, **Pyramid Flow**. These install fine on the Windows cu118/torch-2.2 line *and* the Linux cu12/torch-2.4 line.
- **New-CUDA (12.x) only in practice**: **Wan2.1, HunyuanVideo-I2V, FastHunyuan, LatentSync**. Locked by (1) **bitsandbytes** 4/8-bit CUDA kernels, (2) **flash-attn 2/3** wheels built for the torch-2.4/cu12 line (and the code even refuses to *list* Wan/Hunyuan without it), (3) **diffusers 0.33.1 / transformers 4.56** version-locked classes (`HunyuanVideoTransformer3DModel`, `LlavaForConditionalGeneration`, etc.).
- **Hard-coded CUDA** (won't run CPU/mac regardless of libs): **SyncNet** (`.cuda()` literals).
- (EbSynth's `ebsynth_linux_cu118` native binary was the classic old-CUDA lock, but EbSynth is **dropped** — see Decisions.)

Note: with the **wrap-the-upstream** strategy, these snapshot version numbers are
only the *baseline*. A plugin built on a live upstream lets the installer pick the
**newest** torch/xformers for the detected CUDA line and defers upper bounds to the
upstream — which is exactly how a user on the latest CUDA/torch runs the diffusion
models without hitting the snapshot's ceiling.

---

## Install profiles for the universal installer

Instead of one `requirements.txt`, a plugin declares its **profile bucket**; the
installer resolves the concrete wheels from `(os, gpu, cuda_line)` detected on
the machine (`nvidia-smi` / `checkgpu.py`).

| Bucket | Runs on | torch (target / newest for CUDA line) | Extra | Example modules |
|---|---|---|---|---|
| **A · onnx-CPU** | all OS (mac incl.) | none (or CPU) | `onnxruntime` (mac, opt. CoreML EP) / `onnxruntime-gpu` (win·linux) | Analyser, Separator, FaceSwap, FaceRecognition |
| **B · torch, CPU-ok** | **audio: all OS · visual: win·linux** | mac `2.2.2` CPU (audio only) · win/linux newest for CUDA line | onnxruntime, insightface, spandrel (upscale) | CloneVoice (mac-ok), LivePortrait, GFPGAN, Wav2Lip *(win·linux)* |
| **C · GPU old-CUDA (11.8+)** | win·linux | newest torch for the CUDA line + matching xformers | diffusers, transformers, accelerate; flash-attn optional | SD/ControlNet, OmniGen, Highlights, Pyramid, RealESRGAN(GPU) |
| **D · GPU new-CUDA (12.x)** | **linux** (win only if user supplies flash-attn/bnb) | newest cu12 torch | + **bitsandbytes** + **flash-attn** (cu12) + diffusers | Wan2.1, Hunyuan I2V/T2V, LatentSync |
| **X · CUDA-hardcoded** | win·linux | either | — | SyncNet (`.cuda()` literals) |

(Bucket **E · native binary** removed with EbSynth. SAM-based RemoveBackground is
not a bucket here — it extends the built-in Object Detection.)

### Detection → wheel selection (installer logic)
1. **OS**: `mac` → CPU wheels, **audio buckets only**; refuse C/D, visual bucket B, and any `gpu: req` module.
2. **GPU present?** no NVIDIA → CPU wheels; allow only buckets A/B (CPU paths); mark `gpu: req` modules unavailable → recommend an `api`-kind alternative.
3. **CUDA line** (from driver `CUDA Version` in `nvidia-smi`):
   - `≥12.0` → linux cu12 line (`torch 2.4.0+cu12`, `xformers 0.0.27post2`); buckets A–D OK. (cu118 wheels also run on a 12.x driver — backward compatible — so C is fine here too.)
   - `11.8–11.x` → cu118 line (`torch 2.2.2+cu118`, `xformers 0.0.25.post1+cu118`); buckets A–C OK; **bucket D unavailable** (flash-attn/bnb + diffusers-locked video needs cu12).
   - `<11.8` → treat as CPU or ask the user to update the driver.
4. Compose the venv `requirements`: base (module deps) + the torch/xformers/onnxruntime lines chosen above + bucket extras (diffusers/transformers/bnb/flash-attn as declared). Prefer the official index URLs: `https://download.pytorch.org/whl/cu118` or `.../cu124` (or `cu121`), CPU from PyPI.

### Manifest hooks this implies (extend `plugin.json`)
- `bucket`: `A|B|C|D|X` (or an explicit capability set), so the installer maps to wheels without re-deriving from imports.
- `hardware`: `{ min_vram_gb, cpu_ok }` — already in the format; buckets C/D set `cpu_ok:false`.
- `cuda`: `none | 11.8+ | 12.x` — the minimum CUDA line; installer refuses on a lower line and suggests the API route.
- `os`: subset; installer refuses on an unlisted OS (macOS omitted for C/D, visual B, and SyncNet).
- Keep model weights out of the package (downloaded per `models`), and follow
  the entry/stdout contract in [authoring-guide.md](authoring-guide.md).

### Porting priority (cheapest, widest reach first)
1. **Audio (all OS, CPU-fast):** Separator, CloneVoice — wrap the modern upstreams; these are the macOS story.
2. **Bucket B visual (win·linux GPU):** FaceSwap, LivePortrait, GFPGAN/upscale (via spandrel), Wav2Lip — one plugin each (so a heavy import can't break a light one).
3. **Bucket C (win·linux, old-CUDA):** SD/ControlNet, OmniGen, Highlights, Pyramid Flow, RealESRGAN.
4. **Bucket D (linux, new-CUDA):** Wan2.1, Hunyuan, LatentSync — heaviest; also offer `api`-kind alternatives for machines that can't run them.
5. **Extension, not a plugin:** RemoveBackground on top of the built-in Object Detection (SAM2).
