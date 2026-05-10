<div align="center">

# StorageLLM

**MoE-Native Storage Offloading Engine & JUJU Format**

[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-storagejuju-yellow?style=for-the-badge)](https://huggingface.co/storagejuju)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)
[![Status: Active Development](https://img.shields.io/badge/Status-Active%20Development-orange?style=for-the-badge)]()
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows-lightgrey?style=for-the-badge)]()

<p>
  <a href="#storagellm--moe-native-storage-offloading-engine">English</a> ·
  <a href="#storagellm--moe-전용-스토리지-오프로딩-엔진">한국어</a>
</p>

</div>

---

> **⚠️ Development Build Notice**
> StorageLLM and JUJU format artifacts are currently in active stabilization. PPL and KV-cache correctness validation are in progress. Treat all current artifacts as development builds — runtime behavior and output quality may change as fixes land.

---

## StorageLLM — MoE-Native Storage Offloading Engine

### Overview

StorageLLM is a research and systems project with one core goal: **run large-scale Mixture-of-Experts (MoE) language models on memory-constrained hardware without meaningful quality degradation**. Rather than trying to compress models to fit within available VRAM, StorageLLM treats the storage layer — NVMe SSDs, RAID arrays, or even HDDs — as a first-class memory tier, streaming expert weights on demand with low enough latency that inference remains practical.

The project is built around two tightly coupled components:

1. **The JUJU format** — a custom binary container designed from scratch for high-throughput, Direct I/O–eligible storage streaming in MoE workloads.
2. **The StorageLLM engine** — a C++ inference runtime implementing asynchronous expert residency management across VRAM, RAM, and storage tiers.

---

### Why a New Format? The Case for JUJU

Existing model serialization formats — GGUF, SafeTensors, PyTorch checkpoints — were designed for models that either fit in memory or are loaded once at startup. They make no provisions for partial, on-demand, asynchronous weight streaming, and their internal layouts actively work against efficient Direct I/O, GPU Direct Storage (GDS), or DMA pipelining.

JUJU is built specifically to eliminate these structural inefficiencies for MoE architectures:

#### Physical Layout and Alignment

All tensor payloads in a JUJU file are rewritten into **4096-byte aligned sections**. This is not cosmetic. 4KB alignment is the fundamental requirement for:

- **Linux Direct I/O (`O_DIRECT`)** — bypasses the page cache entirely, eliminating double-buffering of weight data
- **Windows `IORing`** — equivalent zero-copy path on Win32
- **GPU Direct Storage (GDS)** — enables NVMe-to-VRAM transfers without CPU involvement
- **DMA staging** — host-side pinned memory pools (`moe_staging_slot`) require aligned source buffers for maximum throughput

The physical materializer (implemented in `colab/juju_shard_materializer.py` and the Colab notebook) ingests GGUF source shards and rewrites tensor payloads into JUJU sections one shard at a time, keeping Colab disk usage bounded. The output — `<shard>.juju` (immutable weight container) and `<shard>.juju.idx` (runtime index) — is what the StorageLLM engine loads at inference time.

#### The Runtime Index (`.juju.idx`)

The companion index file is not just an offset table. It carries the full runtime planning contract for each tensor/section, including:

- **Residency tier annotations** — which sections belong in VRAM (hot), pinned RAM (warm), or storage (cold) at model load
- **Prefetch and eviction policy hints** — batch-aware scheduling parameters consumed by the engine's expert scheduler
- **QKV packed-cache settings** — layout metadata for the compressed KV cache subsystem
- **Source weight quantization labels** — so the engine knows decompression requirements before initiating a transfer
- **Telemetry hooks** — counters and timestamps for access-pattern learning
- **I/O scheduling directives** — priority and deadline hints for the `io_uring` / `IORing` submission queue

This means the index is a live, mutable document. The engine updates residency statistics and access-interval EMAs back into the index at runtime, allowing the tier scheduler to improve its decisions across inference sessions.

#### The `JUJU_MAX_MOE_STRUCTURE_V1` Contract

The `final_model_structure_contract` embedded in JUJU metadata locks the structural topology that the engine expects:

| Component | Role |
|---|---|
| **Shared Core** | Embedding, normalization, and output projection — always resident in VRAM |
| **Routing Brain** | Token-to-expert router — resident in VRAM, updated per forward pass |
| **Predictor Bank** | Learned expert activation predictor for prefetch scheduling |
| **Expert Groups** | Logical groupings of expert weight shards |
| **Expert Segments** | Physical 4KB-aligned storage segments for individual experts |
| **Buddy Fallback** | Dense fallback path for experts evicted under memory pressure |
| **QKV Subsystem** | Packed key-value cache with optional compression |
| **Coding Locality Cache** | Reuse buffer for experts activated in recent adjacent tokens |
| **Tier Scheduler** | Admission controller for VRAM / RAM / storage promotion and demotion |

Further development targets kernel tuning, predictor weight refinement, and mutable index policy optimization — not changes to this top-level structure.

---

### Engine Architecture

#### 1. Async Storage I/O (`moe_io_ring_adapter`)

StorageLLM integrates natively with platform-native async I/O:

- **Linux**: `io_uring` with submission queue polling for minimal syscall overhead
- **Windows**: `IORing` (Win32 kernel I/O completion API)

Expert weight transfers are issued as Direct I/O requests, bypassing the OS page cache and eliminating the CPU copy that conventional `read()` calls introduce. Active expert weights are transferred directly into VRAM via pinned host staging buffers (`moe_staging_slot`), with GPU event–based synchronization maximizing overlap between compute and I/O.

#### 2. Predictive Expert Residency (`avg_gap_ema`)

The engine tracks per-expert activation intervals using an **exponential moving average of access gaps** (`avg_gap_ema`). This statistic drives:

- **Prefetch scheduling** — experts with a short average gap (frequently activated) are prefetched proactively before they appear on the critical path
- **Eviction ordering** — experts with a long average gap are demoted first when VRAM pressure requires evictions
- **Tier admission control** — automated promotion from storage → RAM → VRAM as access frequency increases, and demotion in the reverse direction as it falls

The predictor bank extends this with a learned component: a lightweight neural predictor trained on expert co-activation patterns, enabling speculative prefetch of experts likely to be needed in upcoming tokens even before the router has selected them.

#### 3. KV Cache Compression

KV cache compression is enabled by default. The QKV subsystem supports packed-cache layouts with configurable quantization, reducing the memory footprint of the attention cache while keeping decompression on the critical path minimal.

---

### Model Artifacts (Hugging Face)

JUJU format artifacts are published to the official Hugging Face organization:

🤗 **[huggingface.co/storagejuju](https://huggingface.co/storagejuju)**

The conversion pipeline (GGUF → JUJU) runs in Google Colab via `colab/GGUF_Offload_Metadata_Patch_Stream.ipynb`. The notebook processes one shard at a time, uploads `<shard>.juju` + `<shard>.juju.idx`, then deletes the local output before advancing to the next shard. Do not use patched GGUF files as the runtime input — the engine expects JUJU artifacts.

---

### Repository Structure

```
storage_llm/
├── colab/                          # Conversion pipeline (GGUF → JUJU)
│   ├── GGUF_Offload_Metadata_Patch_Stream.ipynb   # Main materializer notebook
│   ├── juju_shard_materializer.py  # Core shard rewriter library
│   └── notebooks/                  # Supporting notebooks and docs
├── engine_core/                    # Core runtime (KV, quantization)
├── moe_engine/                     # MoE inference engine (C++)
│   ├── include/moe_pc_engine.h     # Public engine API
│   └── src/                        # Engine implementation
├── benchmarks/                     # Performance benchmarks
├── openclaw.storagellm.json        # OpenAI-compatible local server config
└── LICENSE                         # MIT
```

---

### Quick Start

> Full setup documentation is in progress. The following reflects the current development workflow.

**1. Convert a model (Colab)**

Open `colab/GGUF_Offload_Metadata_Patch_Stream.ipynb` in Google Colab. Set your HuggingFace token and source model URL, then run. The notebook will produce `.juju` + `.juju.idx` shards and upload them to your target HF repo.

**2. Download JUJU artifacts**

```bash
huggingface-cli download storagejuju/<model-name> --local-dir ./models/<model-name>
```

**3. Run inference**

```bash
# OpenAI-compatible API server (see openclaw.storagellm.json for client config)
./storagellm-server --model ./models/<model-name> --port 8000
```

---

### Roadmap

- [ ] Stabilize PPL and KV-cache correctness
- [ ] Complete continuous batching support
- [ ] Hardware async paths (GDS, platform-specific acceleration)
- [ ] Predictor bank training pipeline
- [ ] Comprehensive benchmark suite
- [ ] Full setup and deployment documentation

---

### License

MIT License — see [LICENSE](LICENSE) for details.

---
---

## StorageLLM — MoE 전용 스토리지 오프로딩 엔진

### 프로젝트 개요

StorageLLM은 하나의 핵심 목표를 가진 시스템 연구 프로젝트입니다: **메모리가 제한된 하드웨어에서 대규모 Mixture-of-Experts(MoE) 언어 모델을 품질 저하 없이 실행하는 것**. 모델을 VRAM에 맞게 압축하는 대신, StorageLLM은 NVMe SSD, RAID 어레이, 혹은 HDD까지도 일급 메모리 계층으로 취급합니다. 필요한 Expert 가중치만 온디맨드로 스트리밍하되, 지연 시간을 충분히 낮게 유지하여 실용적인 추론이 가능하도록 합니다.

프로젝트는 두 개의 긴밀하게 결합된 구성요소를 중심으로 구축됩니다:

1. **JUJU 포맷** — MoE 워크로드에서 고처리량, Direct I/O 적격 스토리지 스트리밍을 위해 처음부터 설계된 커스텀 바이너리 컨테이너 포맷
2. **StorageLLM 엔진** — VRAM, RAM, 스토리지 계층 간 비동기 Expert 레지던시 관리를 구현하는 C++ 추론 런타임

---

### 왜 새로운 포맷인가? JUJU의 탄생 배경

GGUF, SafeTensors, PyTorch 체크포인트 등 기존의 모델 직렬화 포맷들은 메모리에 완전히 로드되거나 시작 시 한 번에 적재되는 모델을 위해 설계되었습니다. 이들은 부분적이고 온디맨드적이며 비동기적인 가중치 스트리밍을 전혀 고려하지 않으며, 내부 레이아웃 자체가 Direct I/O, GPU Direct Storage(GDS), DMA 파이프라이닝에 역행하는 구조입니다.

JUJU는 MoE 아키텍처에서 이러한 구조적 비효율을 근본적으로 제거하기 위해 설계되었습니다.

#### 물리적 레이아웃과 정렬

JUJU 파일의 모든 텐서 페이로드는 **4096바이트 정렬 섹션**으로 재기록됩니다. 이는 단순한 형식적 요건이 아닙니다. 4KB 정렬은 다음을 위한 근본적인 요건입니다:

- **Linux Direct I/O (`O_DIRECT`)** — 페이지 캐시를 완전히 우회하여 가중치 데이터의 이중 버퍼링 제거
- **Windows `IORing`** — Win32에서의 동등한 제로-카피 경로
- **GPU Direct Storage (GDS)** — CPU 개입 없는 NVMe→VRAM 직접 전송 활성화
- **DMA 스테이징** — 호스트 측 피닝 메모리 풀(`moe_staging_slot`)이 최대 처리량을 위한 정렬된 소스 버퍼를 필요로 함

물리적 materializer(코랩 노트북 및 `colab/juju_shard_materializer.py`)는 GGUF 소스 샤드를 입력받아 텐서 페이로드를 JUJU 섹션으로 재기록합니다. 한 번에 하나의 샤드씩 처리하여 코랩 디스크 사용량을 제한된 범위 내로 유지합니다. 출력물인 `<샤드>.juju`(불변 가중치 컨테이너)와 `<샤드>.juju.idx`(런타임 인덱스)가 StorageLLM 엔진이 추론 시 로드하는 최종 아티팩트입니다.

#### 런타임 인덱스 (`.juju.idx`)

인덱스 파일은 단순한 오프셋 테이블이 아닙니다. 각 텐서/섹션에 대한 전체 런타임 계획 계약을 포함합니다:

- **레지던시 계층 어노테이션** — 모델 로드 시 각 섹션이 VRAM(핫), 피닝 RAM(웜), 스토리지(콜드) 중 어디에 있어야 하는지
- **프리페치 및 축출 정책 힌트** — 엔진의 Expert 스케줄러가 소비하는 배치 인식 스케줄링 매개변수
- **QKV 패킹 캐시 설정** — 압축 KV 캐시 서브시스템을 위한 레이아웃 메타데이터
- **소스 가중치 양자화 레이블** — 전송 시작 전 엔진이 압축 해제 요건을 파악하기 위한 정보
- **텔레메트리 훅** — 접근 패턴 학습을 위한 카운터 및 타임스탬프
- **I/O 스케줄링 지시사항** — `io_uring`/`IORing` 제출 큐를 위한 우선순위 및 데드라인 힌트

이것은 인덱스가 살아있는 가변 문서임을 의미합니다. 엔진은 런타임에 레지던시 통계와 접근 간격 EMA를 인덱스에 역기록하여, 계층 스케줄러가 추론 세션 전반에 걸쳐 의사결정을 개선할 수 있게 합니다.

#### `JUJU_MAX_MOE_STRUCTURE_V1` 계약

JUJU 메타데이터에 내장된 `final_model_structure_contract`는 엔진이 기대하는 구조적 토폴로지를 확정합니다:

| 구성요소 | 역할 |
|---|---|
| **공유 코어** | 임베딩, 정규화, 출력 투영 — 항상 VRAM에 상주 |
| **라우팅 브레인** | 토큰→Expert 라우터 — VRAM 상주, 순전파마다 업데이트 |
| **예측기 뱅크** | 프리페치 스케줄링을 위한 학습된 Expert 활성화 예측기 |
| **Expert 그룹** | Expert 가중치 샤드의 논리적 그룹화 |
| **Expert 세그먼트** | 개별 Expert를 위한 물리적 4KB 정렬 스토리지 세그먼트 |
| **버디 폴백** | 메모리 압박으로 축출된 Expert를 위한 밀집 폴백 경로 |
| **QKV 서브시스템** | 선택적 압축이 가능한 패킹 키-밸류 캐시 |
| **코딩 지역성 캐시** | 최근 인접 토큰에서 활성화된 Expert를 위한 재사용 버퍼 |
| **계층 스케줄러** | VRAM / RAM / 스토리지 간 승격 및 강등을 위한 입장 제어기 |

이후 개발은 이 최상위 구조 변경 대신 커널 튜닝, 예측기 가중치 개선, 가변 인덱스 정책 최적화에 집중합니다.

---

### 엔진 아키텍처

#### 1. 비동기 스토리지 I/O (`moe_io_ring_adapter`)

StorageLLM은 플랫폼 네이티브 비동기 I/O와 직접 통합됩니다:

- **Linux**: 최소 시스콜 오버헤드를 위한 제출 큐 폴링이 포함된 `io_uring`
- **Windows**: `IORing` (Win32 커널 I/O 완료 API)

Expert 가중치 전송은 OS 페이지 캐시를 우회하는 Direct I/O 요청으로 발행됩니다. 활성 Expert 가중치는 피닝 호스트 스테이징 버퍼(`moe_staging_slot`)를 통해 VRAM으로 직접 전송되며, GPU 이벤트 기반 동기화로 계산과 I/O 간의 오버랩을 최대화합니다.

#### 2. 예측형 Expert 레지던시 관리 (`avg_gap_ema`)

엔진은 **접근 간격의 지수 이동 평균**(`avg_gap_ema`)을 사용하여 Expert별 활성화 간격을 추적합니다. 이 통계는 다음을 구동합니다:

- **프리페치 스케줄링** — 평균 간격이 짧은(자주 활성화되는) Expert를 크리티컬 패스에 나타나기 전에 선제적으로 프리페치
- **축출 우선순위** — VRAM 압박 시 평균 간격이 긴 Expert를 먼저 강등
- **계층 입장 제어** — 접근 빈도 증가에 따라 스토리지 → RAM → VRAM으로 자동 승격, 빈도 감소 시 역방향 강등

예측기 뱅크는 학습된 컴포넌트로 이를 확장합니다: Expert 공동 활성화 패턴으로 학습된 경량 신경 예측기가, 라우터가 Expert를 선택하기 전에도 향후 토큰에서 필요할 가능성이 높은 Expert의 투기적 프리페치를 가능하게 합니다.

#### 3. KV 캐시 압축

KV 캐시 압축은 기본으로 활성화됩니다. QKV 서브시스템은 설정 가능한 양자화가 포함된 패킹 캐시 레이아웃을 지원하여, 크리티컬 패스의 압축 해제 오버헤드를 최소화하면서 어텐션 캐시의 메모리 사용량을 줄입니다.

---

### 모델 아티팩트 (Hugging Face)

JUJU 포맷 아티팩트는 공식 Hugging Face 조직에 공개됩니다:

🤗 **[huggingface.co/storagejuju](https://huggingface.co/storagejuju)**

변환 파이프라인(GGUF → JUJU)은 Google Colab의 `colab/GGUF_Offload_Metadata_Patch_Stream.ipynb`를 통해 실행됩니다. 노트북은 한 번에 하나의 샤드를 처리하고, `<샤드>.juju` + `<샤드>.juju.idx`를 업로드한 후 다음 샤드로 넘어가기 전에 로컬 출력을 삭제합니다. 패치된 GGUF 파일을 런타임 입력으로 사용하지 마세요 — 엔진은 JUJU 아티팩트를 기대합니다.

---

### 현재 상태 및 알려진 이슈

현재 활발한 버그 수정 빌드 중입니다:

- PPL 및 KV 캐시 정확도 검증이 진행 중
- 연속 배칭, 일부 하드웨어 비동기 경로, 플랫폼별 가속 경로가 미완성이거나 재작업 중
- 모델 아티팩트는 로드될 수 있으나, 엔진 안정화 중에는 출력 품질과 속도가 변경될 수 있음

현재 아티팩트는 개발 빌드로 취급하세요. 런타임 및 포맷 수정이 수일 내로 릴리스될 예정입니다.

---

### 라이선스

MIT 라이선스 — 자세한 내용은 [LICENSE](LICENSE)를 참조하세요.
