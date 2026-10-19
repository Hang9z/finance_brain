import argparse
import sys
import time
import uuid
from pathlib import Path

# 确保脚本运行时正确加载工程根目录模块
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from common.logging.logger import logger
from processor.import_processor.main_graph import kb_import_app
from processor.import_processor.state import create_default_state
from utils.path_util import PROJECT_ROOT

DEFAULT_DATA_DIR = Path(r"D:\尚硅谷笔记\阶段八：尚硅谷大模型技术之掌柜智库项目\尚硅谷大模型项目实战之掌柜智库实战\1.资料\金融\数据")


def scan_financial_files(data_dir: Path) -> list[Path]:
    """
    扫描金融数据目录下所有支持格式的文件 (PDF, DOCX, DOC)
    """
    if not data_dir.exists() or not data_dir.is_dir():
        logger.error(f"金融数据目录不存在: {data_dir}")
        return []

    supported_suffixes = {".pdf", ".docx", ".doc"}
    files = [
        f for f in data_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in supported_suffixes and not f.name.startswith("~$")
    ]
    # 按目录与文件名稳定排序
    files.sort(key=lambda p: (p.parent.name, p.name))
    return files


def import_single_file(file_path: Path, output_dir: Path, dry_run: bool = False) -> dict:
    """
    执行单个金融文件的全链路结构化入库流程
    """
    file_title = file_path.stem
    category = file_path.parent.name
    logger.info(f"\n========================================================")
    logger.info(f"开始处理 [{category}] 资料: {file_path.name}")
    logger.info(f"========================================================")

    if dry_run:
        logger.info(f"[DRY-RUN] 模拟执行，跳过实际图执行与向量化入库: {file_path.name}")
        return {"file": file_path.name, "category": category, "status": "dry_run", "chunks": 0}

    task_id = f"fin_{uuid.uuid4().hex[:8]}"
    start_time = time.time()

    try:
        init_state = create_default_state(
            task_id=task_id,
            local_file_path=str(file_path),
            local_dir=str(output_dir),
            file_title=file_title
        )

        # 执行 LangGraph 导入流水线
        result_state = kb_import_app.invoke(init_state)

        elapsed = time.time() - start_time
        chunks_count = len(result_state.get("chunks", []))
        item_name = result_state.get("item_name", "未知")

        logger.success(
            f"文件【{file_path.name}】入库完成! 耗时: {elapsed:.2f}s | "
            f"识别实体: 【{item_name}】 | 生成切片数: {chunks_count}"
        )
        return {
            "file": file_path.name,
            "category": category,
            "status": "success",
            "item_name": item_name,
            "chunks": chunks_count,
            "elapsed": elapsed
        }

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"文件【{file_path.name}】导入失败! 耗时: {elapsed:.2f}s, 错误: {e}", exc_info=True)
        return {
            "file": file_path.name,
            "category": category,
            "status": "failed",
            "error": str(e),
            "elapsed": elapsed
        }


def main():
    parser = argparse.ArgumentParser(description="金融知识库 17 份核心资料批量导入工具")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR), help="金融数据根目录")
    parser.add_argument("--category", type=str, default=None, help="仅导入指定子分类目录（如 '上市公司年报'）")
    parser.add_argument("--file", type=str, default=None, help="仅导入指定文件名（模糊匹配）")
    parser.add_argument("--dry-run", action="store_true", help="演练模式，仅扫描与校验文件，不触发实际嵌入入库")
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "output"), help="输出目录")

    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("========== 金融知识库资料自动化导入启动 ==========")
    logger.info(f"数据根目录: {data_dir}")
    logger.info(f"规整输出目录: {output_dir}")
    logger.info(f"演练模式 (dry-run): {args.dry_run}")

    all_files = scan_financial_files(data_dir)
    if not all_files:
        logger.warning("未扫描到任何待导入的金融文件，流程结束。")
        return

    # 过滤条件
    target_files = all_files
    if args.category:
        target_files = [f for f in target_files if args.category in f.parent.name]
    if args.file:
        target_files = [f for f in target_files if args.file.lower() in f.name.lower()]

    logger.info(f"共发现 {len(all_files)} 份资料，经筛选后待处理 {len(target_files)} 份：")
    for idx, f in enumerate(target_files, start=1):
        logger.info(f"  [{idx:02d}] [{f.parent.name}] {f.name} ({f.suffix.upper()})")

    results = []
    total_start = time.time()

    for idx, file_path in enumerate(target_files, start=1):
        logger.info(f"\n>>> 进度 [{idx}/{len(target_files)}] 准备导入: {file_path.name}")
        res = import_single_file(file_path, output_dir, dry_run=args.dry_run)
        results.append(res)

    total_elapsed = time.time() - total_start
    success_count = sum(1 for r in results if r.get("status") == "success")
    failed_count = sum(1 for r in results if r.get("status") == "failed")
    dry_count = sum(1 for r in results if r.get("status") == "dry_run")
    total_chunks = sum(r.get("chunks", 0) for r in results)

    logger.info("\n" + "=" * 60)
    logger.info("========== 金融知识库批量导入汇总统计 ==========")
    logger.info(f"总处理文档: {len(results)} 份")
    logger.info(f"成功: {success_count} 份 | 失败: {failed_count} 份 | 演练: {dry_count} 份")
    logger.info(f"入库总切片数: {total_chunks} 个")
    logger.info(f"总耗时: {total_elapsed:.2f} 秒")
    logger.info("=" * 60 + "\n")


if __name__ == "__main__":
    main()
