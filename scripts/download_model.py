import argparse
import os
import sys
from pathlib import Path


# 项目所需模型的下载命令：
# python scripts/download_model.py Qwen/Qwen3-1.7B-Base
# python scripts/download_model.py Qwen/Qwen3-0.6B-Base
# 默认使用 ModelScope 国内高速源，支持指定 Hugging Face/HF-Mirror
# 默认落盘路径: <项目根目录>/assets/models/<模型名称>/

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_BASE = PROJECT_ROOT / "assets" / "models"


def main():
    parser = argparse.ArgumentParser(
        description="Download a model repository from ModelScope or Hugging Face Hub."
    )
    parser.add_argument(
        "repo_id",
        type=str,
        help="模型仓库 ID，例如 Qwen/Qwen3-1.7B-Base",
    )
    parser.add_argument(
        "--source",
        choices=["modelscope", "ms", "huggingface", "hf"],
        default="modelscope",
        help="下载源：'modelscope'/'ms' (国内推荐高速) 或 'huggingface'/'hf'",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="模型保存目录；默认是 assets/models/<仓库名称>",
    )
    parser.add_argument(
        "--revision",
        default="master",
        help="仓库的分支、标签或 commit；ModelScope 默认为 master，HF 建议设为 main",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        model_name = args.repo_id.rstrip("/").split("/")[-1]
        output_dir = DEFAULT_MODELS_BASE / model_name
    else:
        output_dir = args.output_dir
        if not output_dir.is_absolute():
            output_dir = Path.cwd() / output_dir

    print(f"Model: {args.repo_id}")
    print(f"Source: {args.source}")
    print(f"Revision: {args.revision}")
    print(f"Save to: {output_dir}")

    if args.source in ["modelscope", "ms"]:
        try:
            from modelscope import snapshot_download
        except ImportError:
            print("\n[错误] 未检测到 modelscope，请先安装: pip install modelscope\n")
            sys.exit(1)

        snapshot_download(
            model_id=args.repo_id,
            revision=args.revision,
            local_dir=str(output_dir),
        )
    else:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            print("\n[错误] 未检测到 huggingface_hub，请先安装: pip install huggingface_hub\n")
            sys.exit(1)

        # 国内镜像加速配置
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        
        # 若用户没手动指定 revision 且仍为默认 master，自动转为 HF 默认的 main
        revision = "main" if args.revision == "master" else args.revision

        snapshot_download(
            repo_id=args.repo_id,
            revision=revision,
            local_dir=str(output_dir),
            local_dir_use_symlinks=False,  # 确保实际权重文件实体落盘
        )

    # 简单校验关键文件是否存在
    has_config = (output_dir / "config.json").exists()
    has_weights = any(output_dir.glob("*.safetensors")) or any(output_dir.glob("*.bin"))
    
    if has_config and has_weights:
        print("✅ Download and verification complete.")
    else:
        print("⚠️ Download finished, but config.json or weight files might be missing.")


if __name__ == "__main__":
    main()