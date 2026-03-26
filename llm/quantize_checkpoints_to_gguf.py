import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], cwd: Path | None = None):
    p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=False)
    if p.returncode != 0:
        raise RuntimeError(f"命令执行失败: {cmd} (exit={p.returncode})")


def _find_quantize_bin(
    llama_cpp_dir: Path, build_dir: Path, build_type: str
) -> Path | None:
    names = (
        "llama-quantize",
        "llama-quantize.exe",
        "quantize",
        "quantize.exe",
    )
    candidates: list[Path] = []
    for name in names:
        candidates.append(llama_cpp_dir / name)
        candidates.append(llama_cpp_dir / "build" / "bin" / name)
        candidates.append(llama_cpp_dir / "build" / "bin" / build_type / name)
        candidates.append(build_dir / "bin" / name)
        candidates.append(build_dir / "bin" / build_type / name)
        candidates.append(build_dir / name)
        candidates.append(build_dir / build_type / name)
    for p in candidates:
        if p.exists():
            return p
    return None


def _maybe_build_llama_cpp(llama_cpp_dir: Path, build_dir: Path, build_type: str):
    cmake = os.environ.get("CMAKE", "cmake")
    build_dir.mkdir(parents=True, exist_ok=True)
    _run(
        [
            cmake,
            "-S",
            str(llama_cpp_dir),
            "-B",
            str(build_dir),
            f"-DCMAKE_BUILD_TYPE={build_type}",
        ]
    )
    _run([cmake, "--build", str(build_dir), "--config", build_type])


def _collect_hf_model_dirs(checkpoints_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    if (checkpoints_dir / "config.json").exists():
        candidates.append(checkpoints_dir)
    for d in checkpoints_dir.iterdir():
        if d.is_dir() and (d / "config.json").exists():
            candidates.append(d)
    return candidates


def _maybe_disable_incomplete_index(model_dir: Path) -> Path | None:
    merged = model_dir / "model.safetensors"
    index_path = model_dir / "model.safetensors.index.json"
    if not merged.exists():
        return None
    if not index_path.exists():
        return None
    try:
        obj = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = obj.get("weight_map", None)
        if not isinstance(weight_map, dict) or not weight_map:
            return None
        missing = []
        for part in sorted({str(v) for v in weight_map.values()}):
            if not (model_dir / part).exists():
                missing.append(part)
        if not missing:
            return None
        bak = model_dir / "model.safetensors.index.json.bak_missing_shards"
        try:
            bak.unlink(missing_ok=True)
        except Exception:
            pass
        index_path.replace(bak)
        print(
            f"提示: {model_dir} 的 model.safetensors.index.json 引用了缺失分片，已临时忽略 index，使用单文件 model.safetensors"
        )
        return bak
    except Exception:
        return None


def _restore_index(model_dir: Path, bak: Path | None) -> None:
    if bak is None:
        return
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        return
    if bak.exists():
        bak.replace(index_path)


def _convert_hf_to_gguf(
    llama_cpp_dir: Path,
    model_dir: Path,
    out_gguf: Path,
    outtype: str,
):
    convert_py = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not convert_py.exists():
        raise FileNotFoundError(f"未找到: {convert_py}")
    out_gguf.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            sys.executable,
            str(convert_py),
            str(model_dir),
            "--outfile",
            str(out_gguf),
            "--outtype",
            str(outtype),
        ],
        cwd=llama_cpp_dir,
    )


def _quantize(
    quantize_bin: Path,
    in_gguf: Path,
    out_gguf: Path,
    qtype: str,
    nthread: int | None,
):
    out_gguf.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(quantize_bin), str(in_gguf), str(out_gguf), str(qtype)]
    if isinstance(nthread, int) and nthread > 0:
        cmd.append(str(nthread))
    _run(cmd, cwd=quantize_bin.parent)


def main(argv: list[str] | None = None):
    p = argparse.ArgumentParser()
    p.add_argument(
        "--llama_cpp_dir",
        default=str(Path(__file__).resolve().parent / "llama.cpp"),
        help="llama.cpp 源码目录 (默认使用当前目录内的 llama.cpp)",
    )
    p.add_argument(
        "--checkpoints_dir",
        default=str(Path(__file__).resolve().parent / "checkpoints"),
        help="输入 checkpoints 目录 (会自动扫描其中的 HF 模型目录)",
    )
    p.add_argument(
        "--build",
        action="store_true",
        help="若 quantize 可执行文件不存在，则尝试构建 llama.cpp",
    )
    p.add_argument(
        "--build_dir",
        default="",
        help="自定义 build 目录 (默认: <llama_cpp_dir>/build)",
    )
    p.add_argument(
        "--build_type",
        default="Release",
        choices=["Release", "RelWithDebInfo", "Debug"],
        help="CMake 构建类型",
    )
    p.add_argument(
        "--outtype",
        default="f16",
        choices=["f16", "f32"],
        help="先转换为 GGUF 的精度",
    )
    p.add_argument(
        "--qtype",
        default="Q4_K_M",
        help="量化类型 (默认 Q4_K_M)",
    )
    p.add_argument(
        "--nthread",
        type=int,
        default=0,
        help="quantize 线程数 (0 表示不传该参数)",
    )
    p.add_argument(
        "--model",
        default="",
        help="只量化指定模型目录名 (位于 checkpoints_dir 下)，为空表示扫描全部",
    )
    args = p.parse_args(argv)

    llama_cpp_dir = Path(args.llama_cpp_dir).expanduser().resolve()
    checkpoints_dir = Path(args.checkpoints_dir).expanduser().resolve()
    build_dir = (
        Path(args.build_dir).expanduser().resolve()
        if str(args.build_dir).strip()
        else (llama_cpp_dir / "build")
    )

    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"找不到 checkpoints_dir: {checkpoints_dir}")

    model_dirs = _collect_hf_model_dirs(checkpoints_dir)
    if str(args.model).strip():
        target = checkpoints_dir / str(args.model).strip()
        model_dirs = [d for d in model_dirs if d.resolve() == target.resolve()]

    if not model_dirs:
        raise FileNotFoundError(
            f"在 {checkpoints_dir} 下未找到 HuggingFace 模型目录 (缺少 config.json)。\n"
            "请把 HF 模型目录放到 checkpoints 下，例如:\n"
            "checkpoints/<model_name>/config.json + tokenizer.* + *.safetensors"
        )

    quantize_bin = _find_quantize_bin(llama_cpp_dir, build_dir, args.build_type)
    if quantize_bin is None and args.build:
        _maybe_build_llama_cpp(llama_cpp_dir, build_dir, args.build_type)
        quantize_bin = _find_quantize_bin(llama_cpp_dir, build_dir, args.build_type)
    if quantize_bin is None:
        raise FileNotFoundError(
            "未找到 llama.cpp 的 quantize 可执行文件。\n"
            "请先构建 llama.cpp，或加 --build 自动构建。\n"
            f"llama_cpp_dir={llama_cpp_dir}\n"
            f"build_dir={build_dir}"
        )

    tmp_dir = checkpoints_dir / ".gguf_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for model_dir in model_dirs:
        name = model_dir.name if model_dir != checkpoints_dir else "checkpoints"
        inter = tmp_dir / f"{name}.{args.outtype}.gguf"
        out = checkpoints_dir / f"{name}.{args.qtype}.gguf"
        bak = _maybe_disable_incomplete_index(model_dir)
        try:
            _convert_hf_to_gguf(llama_cpp_dir, model_dir, inter, args.outtype)
        finally:
            _restore_index(model_dir, bak)
        _quantize(
            quantize_bin=quantize_bin,
            in_gguf=inter,
            out_gguf=out,
            qtype=args.qtype,
            nthread=(args.nthread if args.nthread > 0 else None),
        )
        try:
            inter.unlink(missing_ok=True)
        except Exception:
            pass
        print(f"完成: {out}")

    try:
        if tmp_dir.exists() and not any(tmp_dir.iterdir()):
            tmp_dir.rmdir()
    except Exception:
        pass


if __name__ == "__main__":
    main()
