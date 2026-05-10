from __future__ import annotations
import argparse, importlib.util, json, os, subprocess, sys
from pathlib import Path
SUPPORTED_BACKENDS = ["auto", "cpu", "cuda", "metal", "vulkan", "opencl", "rocm", "hip", "sycl"]
def _has_tvm() -> tuple[bool, str]:
    spec = importlib.util.find_spec("tvm")
    if spec is None: return False, ""
    try:
        import tvm  # type: ignore
        return True, getattr(tvm, "__version__", "unknown")
    except Exception as exc:
        return False, f"import failed: {exc}"
def _candidate_packages(backend: str) -> list[str]:
    forced = os.environ.get("STORAGELLM_TVM_PIP_PACKAGE", "").strip()
    if forced: return [forced]
    backend = (backend or "auto").lower()
    if backend == "cuda": return ["mlc-ai-nightly-cu128", "mlc-ai-nightly-cu121", "mlc-ai-nightly-cpu", "apache-tvm"]
    if backend in {"rocm", "hip"}: return ["mlc-ai-nightly-rocm", "mlc-ai-nightly-cpu", "apache-tvm"]
    return ["mlc-ai-nightly-cpu", "apache-tvm"]
def install_tvm(backend: str, dry_run: bool = False) -> dict:
    ok, version = _has_tvm()
    if ok: return {"ok": True, "already_installed": True, "version": version, "package": None, "attempts": []}
    attempts=[]
    if os.environ.get("STORAGELLM_TVM_AUTO_INSTALL", "1").lower() in {"0","false","no","off"}:
        return {"ok": False, "already_installed": False, "reason": "STORAGELLM_TVM_AUTO_INSTALL=0", "attempts": attempts}
    for pkg in _candidate_packages(backend):
        cmd=[sys.executable,"-m","pip","install","-U",pkg]
        if dry_run:
            attempts.append({"package":pkg,"cmd":cmd,"returncode":None,"dry_run":True}); continue
        try:
            proc=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=int(os.environ.get("STORAGELLM_PIP_TIMEOUT_SEC","180")))
            attempts.append({"package":pkg,"cmd":cmd,"returncode":proc.returncode,"log_tail":proc.stdout[-4000:]})
        except subprocess.TimeoutExpired as exc:
            attempts.append({"package":pkg,"cmd":cmd,"returncode":124,"log_tail":((exc.stdout or "") if isinstance(exc.stdout,str) else "")[-4000:],"timeout":True})
            continue
        ok, version = _has_tvm()
        if proc.returncode == 0 and ok: return {"ok": True, "already_installed": False, "version": version, "package": pkg, "attempts": attempts}
    ok, version = _has_tvm()
    return {"ok": ok, "already_installed": False, "version": version, "package": None, "attempts": attempts}
def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("--backend",default="auto",choices=SUPPORTED_BACKENDS); ap.add_argument("--dry-run",action="store_true"); ap.add_argument("--status-out",default="build/tvm_codegen/tvm_dependency_status.json")
    args=ap.parse_args(); status=install_tvm(args.backend,args.dry_run); out=Path(args.status_out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(status,indent=2),encoding="utf-8"); print(json.dumps(status,indent=2)); return 0 if status.get("ok") or args.dry_run else 2
if __name__ == "__main__": raise SystemExit(main())
