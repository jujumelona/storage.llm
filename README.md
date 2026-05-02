[English](#storagellm-moe-native-storage-offloading-engine) | [한국어](#moe-전용-스토리지-오프로딩-엔진-구현-상세)

# StorageLLM: MoE-Native Storage Offloading Engine

[Hugging Face Repository](https://huggingface.co/storagejuju) | License: MIT

## Project Goal
Storage-based offloading architecture for running large-scale MoE models in memory-constrained environments.

## JUJU Format and Native Offloading
StorageLLM implements an expert-granularity offloading strategy, streaming required weights directly from storage.
- **JUJU Format**: Specialized format optimized for high-throughput storage streaming and asynchronous execution.
- **Hierarchical Memory Management**: Orchestrates data movement between VRAM, RAM, and Storage tiers to control the active memory footprint.
- **Compressed KV Cache**: KV cache compression is enabled by default.

## Model Sources
Download the required JUJU format artifacts from the official repository:
- **Hugging Face**: [https://huggingface.co/storagejuju](https://huggingface.co/storagejuju)

## Specialized MoE Offloading Architecture

### 1. High-Performance Async Storage I/O
- **`moe_io_ring_adapter`**: Native integration with Linux `io_uring` and Windows `IORing` for low-overhead disk access.
- **Expert-Streaming Pipeline**: Direct-to-VRAM transfers for active experts, minimizing CPU-GPU synchronization bottlenecks.

### 2. Predictive Expert Residency Management
- **`avg_gap_ema` (Access Interval EMA)**: Tracks expert activation patterns to predict and prefetch required weights.
- **Tiered Admission Control**: Automated promotion and demotion of experts across VRAM, RAM, and Storage tiers based on access frequency.

### 3. Optimized DMA Transfer Pipeline
- **`moe_staging_slot`**: Pinned host memory pool for high-throughput DMA transfers.
- **Event-based Synchronization**: Per-slot synchronization using GPU events to maximize compute and I/O overlap.

---

## MoE 전용 스토리지 오프로딩 엔진 구현 상세

[Hugging Face 저장소](https://huggingface.co/storagejuju) | 라이선스: MIT

### 프로젝트 목표
저사양 하드웨어 환경에서 초대형 MoE 모델 구동을 위한 스토리지 기반 오프로딩 아키텍처

### 모델 소스
공식 저장소에서 JUJU 포맷 아티팩트를 다운로드할 수 있습니다:
- **Hugging Face**: [https://huggingface.co/storagejuju](https://huggingface.co/storagejuju)

### 핵심 구현 사항 (Code-Verified)
- **스토리지 직결 I/O**: `io_uring` 및 `IORing`을 사용하여 전문가 가중치를 실시간으로 스트리밍합니다.
- **계층적 메모리 관리**: VRAM, RAM, 스토리지를 계층적으로 활용하여 물리적 메모리 제약을 극복합니다.
- **예측형 레지던시 관리**: `avg_gap_ema` 알고리즘으로 전문가의 접근 패턴을 학습하고 비동기 프리페칭을 수행합니다.
- **기본 KV 캐시 압축**: 별도의 설정 없이 KV 캐시 압축이 기본으로 활성화되어 동작합니다.

## License
This project is released under the MIT License.
