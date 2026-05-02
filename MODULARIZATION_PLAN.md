# 📦 모듈화 계획 (Modularization Plan)

## 🎯 목적
큰 파일들을 논리적 단위로 분리하여 유지보수성과 가독성 향상

---

## 📊 현재 상태

### Top 15 Largest Files
1. **model_root_validate.cpp.inc**: 2011 lines ⚠️ CRITICAL
2. **codec_table.cpp.inc**: 897 lines ⚠️ HIGH
3. **juju_runtime_strategy.cpp.inc**: 510 lines ⚠️ HIGH
4. **io_worker_loop.cpp.inc**: 472 lines ⚠️ MEDIUM
5. **staging_slots.cpp.inc**: 419 lines ⚠️ MEDIUM
6. **runtime_orchestrator.cpp.inc**: 368 lines
7. **prefetch_bytes.cpp.inc**: 343 lines
8. **common_raw_prefetch.cpp.inc**: 338 lines
9. **generation_attention.cpp.inc**: 317 lines
10. **engine_state.cpp.inc**: 294 lines
11. **tensor_paths_mmap.cpp.inc**: 292 lines
12. **expert_prefetch_api.cpp.inc**: 291 lines
13. **tensor_dot_cpu_kernels.cpp.inc**: 278 lines
14. **juju_format.cpp.inc**: 268 lines
15. **generation_api_main.cpp.inc**: 265 lines

---

## 🔥 우선순위 1: model_root_validate.cpp.inc (2011 lines)

### 현재 구조 분석 필요
- JSON parsing
- Model validation
- Path resolution
- GGUF/JUJU format handling

### 제안 분리
```
model_root_validate.cpp.inc (2011 lines)
├─ model_root_validate_core.cpp.inc (main validation logic)
├─ model_root_json_parser.cpp.inc (JSON parsing)
├─ model_root_path_resolver.cpp.inc (path resolution)
├─ model_root_gguf_handler.cpp.inc (GGUF format)
└─ model_root_juju_handler.cpp.inc (JUJU format)
```

---

## 🔥 우선순위 2: codec_table.cpp.inc (897 lines)

### 현재 구조 분석 필요
- CSV parsing
- Binary index loading
- GGUF tensor directory
- JUJU tensor index

### 제안 분리
```
codec_table.cpp.inc (897 lines)
├─ codec_table_core.cpp.inc (main table logic)
├─ codec_table_csv.cpp.inc (CSV parsing)
├─ codec_table_binary.cpp.inc (binary index)
├─ codec_table_gguf.cpp.inc (GGUF directory)
└─ codec_table_juju.cpp.inc (JUJU index)
```

---

## 🔥 우선순위 3: juju_runtime_strategy.cpp.inc (510 lines)

### 현재 구조 분석 필요
- Runtime strategy selection
- Configuration parsing
- Performance tuning

### 제안 분리
```
juju_runtime_strategy.cpp.inc (510 lines)
├─ juju_strategy_core.cpp.inc (main strategy logic)
├─ juju_strategy_config.cpp.inc (configuration)
└─ juju_strategy_tuning.cpp.inc (performance tuning)
```

---

## 📋 모듈화 원칙

### 1. 단일 책임 원칙 (Single Responsibility)
- 각 파일은 하나의 명확한 책임만 가짐
- 예: JSON parsing, CSV parsing, validation 등

### 2. 응집도 최대화 (High Cohesion)
- 관련된 함수들을 같은 파일에 배치
- 예: 모든 JSON 관련 함수는 json_parser.cpp.inc에

### 3. 결합도 최소화 (Low Coupling)
- 파일 간 의존성 최소화
- 명확한 인터페이스 정의

### 4. 파일 크기 제한
- **목표**: 300 lines 이하
- **최대**: 500 lines
- **경고**: 500+ lines

### 5. 명명 규칙
```
<module>_<submodule>_<function>.cpp.inc

예시:
- model_root_validate_core.cpp.inc
- model_root_json_parser.cpp.inc
- codec_table_csv.cpp.inc
```

---

## 🔧 모듈화 프로세스

### Phase 1: 분석 (Analysis)
1. 파일 읽기
2. 함수 목록 추출
3. 함수 간 의존성 분석
4. 논리적 그룹 식별

### Phase 2: 설계 (Design)
1. 모듈 구조 설계
2. 인터페이스 정의
3. 의존성 그래프 작성

### Phase 3: 구현 (Implementation)
1. 새 파일 생성
2. 함수 이동
3. 인터페이스 구현
4. 컴파일 검증

### Phase 4: 검증 (Verification)
1. 컴파일 테스트
2. 기능 테스트
3. 성능 테스트
4. 문서 업데이트

---

## 📈 예상 효과

### 유지보수성
- **Before**: 2011 lines 파일 → 수정 어려움
- **After**: 5개 × 400 lines → 수정 용이

### 가독성
- **Before**: 스크롤 많이 필요
- **After**: 관련 코드만 보면 됨

### 컴파일 시간
- **Before**: 큰 파일 수정 시 전체 재컴파일
- **After**: 작은 파일 수정 시 부분 재컴파일

### 협업
- **Before**: 동시 수정 시 conflict 많음
- **After**: 다른 모듈 수정 시 conflict 적음

---

## 🚀 실행 계획

### Week 1: model_root_validate.cpp.inc
- Day 1-2: 분석 및 설계
- Day 3-4: 구현
- Day 5: 검증 및 문서화

### Week 2: codec_table.cpp.inc
- Day 1-2: 분석 및 설계
- Day 3-4: 구현
- Day 5: 검증 및 문서화

### Week 3: juju_runtime_strategy.cpp.inc
- Day 1-2: 분석 및 설계
- Day 3-4: 구현
- Day 5: 검증 및 문서화

### Week 4: 나머지 파일들
- io_worker_loop.cpp.inc
- staging_slots.cpp.inc
- 기타 필요한 파일들

---

## ⚠️ 주의사항

### 1. 기능 변경 금지
- 모듈화는 **구조 변경**만
- 기능은 **절대 변경하지 않음**
- 모든 버그픽스는 **별도 작업**

### 2. 컴파일 검증 필수
- 각 단계마다 컴파일 확인
- 모든 warning 해결
- 기능 테스트 통과

### 3. 문서화 필수
- 각 모듈의 역할 명시
- 인터페이스 문서화
- 의존성 그래프 작성

### 4. 점진적 진행
- 한 번에 하나씩
- 검증 후 다음 단계
- 문제 발생 시 즉시 rollback

---

## 📚 참고 자료

### 모듈화 Best Practices
1. Clean Code (Robert C. Martin)
2. Code Complete (Steve McConnell)
3. Refactoring (Martin Fowler)

### C++ Specific
1. Large-Scale C++ Software Design (John Lakos)
2. C++ Coding Standards (Sutter & Alexandrescu)

---

## ✅ 체크리스트

### 모듈화 전
- [ ] 파일 분석 완료
- [ ] 함수 목록 추출
- [ ] 의존성 분석 완료
- [ ] 모듈 구조 설계 완료

### 모듈화 중
- [ ] 새 파일 생성
- [ ] 함수 이동
- [ ] 인터페이스 구현
- [ ] 컴파일 성공

### 모듈화 후
- [ ] 기능 테스트 통과
- [ ] 성능 테스트 통과
- [ ] 문서 업데이트
- [ ] Code review 완료

---

## 🎯 목표

**모든 파일을 500 lines 이하로 유지하여 유지보수성과 가독성을 극대화!**

현재: 2011 lines (최대)
목표: 500 lines (최대)
**개선: 75% 감소**
