import os
import shutil
import time
from pathlib import Path
from typing import Tuple, List

import fitz
import requests

from common.config.mineru_config import mineru_config
from common.logging.logger import logger, node_log, step_log
from processor.import_processor.state import ImportGraphState, create_default_state
from utils.path_util import PROJECT_ROOT
from utils.task_utils import add_running_task, add_done_task

# MinerU 单次请求最大页数限制（默认 200 页，支持环境变量覆盖）
MAX_PAGES_PER_BATCH = int(os.getenv("MINERU_MAX_PAGES_PER_BATCH", 200))


def get_pdf_page_count(pdf_path: Path) -> int:
    """获取 PDF 文件总页数"""
    try:
        with fitz.open(pdf_path) as doc:
            return len(doc)
    except Exception as e:
        logger.warning(f"[node_pdf_to_md] 读取 PDF 页数异常，默认回退单卷模式: {e}")
        return 1

@node_log("node_pdf_to_md")
def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:
    """
    节点: PDF转Markdown (node_pdf_to_md)
    核心任务:
    1. 校验 PDF 文件有效性与页数。
    2. 若总页数 <= 200 页，直接调用 MinerU 统一解析。
    3. 若总页数 > 200 页，自动切分为多个分卷 PDF（每卷 <= 200 页），分卷提交 MinerU 解析后无缝合并。
    4. 将结果保存到 state["md_path"]。
    """
    # 1. 任务状态追踪记录
    add_running_task(state["task_id"], "node_pdf_to_md")

    # 2. 校验路径和物理文件
    pdf_path_obj, local_dir_obj = step_1_validate_and_get_data(state)

    # 3. 页数校验与智能路由
    total_pages = get_pdf_page_count(pdf_path_obj)
    if total_pages > MAX_PAGES_PER_BATCH:
        logger.info(f"[node_pdf_to_md] 文档 [{pdf_path_obj.name}] 共 {total_pages} 页，超过单次解析限制 ({MAX_PAGES_PER_BATCH} 页)，启用自动分卷解析流水线")
        md_path = step_split_upload_and_merge(pdf_path_obj, local_dir_obj, total_pages, MAX_PAGES_PER_BATCH)
    else:
        # 单卷标准解析
        zip_url = step_2_upload_and_poll(pdf_path_obj)
        md_path = step_3_download_and_extract(zip_url, local_dir_obj, pdf_path_obj.stem)

    # 4. 更新状态
    state["md_path"] = str(md_path)

    # 5. 更新任务状态
    add_done_task(state["task_id"], "node_pdf_to_md")
    return state

@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state: ImportGraphState) -> Tuple[Path, Path]:
    """
    步骤一: 校验路径与目录初始化
    :param: state
    :return: pdf_path_obj [Path], local_dir_ob [Path]
    """
    # 1. 获取 state 中的地址
    pdf_path = state.get("pdf_path")
    local_dir = state.get("local_dir")

    # 2. 做非空校验与默认值兜底
    if not pdf_path:
        logger.error("pdf_path的值为空, 无法读取文件, 直接抛出异常")
        raise ValueError("pdf_path的参数值为空, 无法读取文件")

    if not local_dir:
        logger.warning("没有传入local_dir地址, 给予默认值: PROJECT_ROOT / 'output'")
        local_dir = PROJECT_ROOT / "output"
        state["local_dir"] = str(local_dir)

    # 3. 将地址转换成 Path 对象
    pdf_path_obj = Path(pdf_path)
    local_dir_obj = Path(local_dir)

    # 4. 判断物理路径是否合规
    if not pdf_path_obj.is_file():
        logger.error(f"pdf_path: {pdf_path_obj} 不存在或者不是一个文件")
        raise ValueError(f"pdf_path: {pdf_path_obj} 不存在或者不是一个文件")

    if not local_dir_obj.is_dir():
        logger.warning(f"local_dir: {local_dir_obj} 不存在或不是文件夹, 主动创建")
        local_dir_obj.mkdir(parents=True, exist_ok=True)

    # 5. 返回规范化的两个 Path 对象
    return pdf_path_obj, local_dir_obj

@step_log("step_2_upload_and_poll")
def step_2_upload_and_poll(pdf_path_boj: Path) -> str:
    """
    步骤二: 上传文件到 MinerU 并轮询等待解析完成
    :param pdf_path_boj
    :return: zip_url
    """
    # 1. MinerU配置参数
    if not mineru_config.base_url or not mineru_config.api_key:
        logger.error("MinerU配置错误, 请检查 .env中的 MINERU_BASE_URL 与 MINERU_API_TOKEN")
        raise ValueError("MinerU配置错误, 请检查 MinerU 配置")

    # 2. 申请上传地址和批量 ID
    token = mineru_config.api_key
    url = f"{mineru_config.base_url}/file-urls/batch"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }

    data = {
        "files": [
            {"name": pdf_path_boj.name, "data_id": pdf_path_boj.stem}
        ],
        "model_version": "vlm"
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code != 200:
        logger.error(f"申请上传地址失败, HTTP状态码: {response.status_code}")
        raise RuntimeError(f"申请上传地址失败, HTTP状态码: {response.status_code}")

    response_dict = response.json()
    if response_dict.get("code") != 0:
        logger.error(f"申请上传地址业务失败! 错误码: {response_dict.get('code')}, 消息:{response_dict.get('msg')}")
        raise RuntimeError(f"申请上传地址业务失败: {response_dict.get('msg')}")

    file_upload_urls = response_dict["data"].get("file_urls", [])
    batch_id = response_dict["data"].get("batch_id")
    if not file_upload_urls or not batch_id:
        logger.error("[node_pdf_to_md] MinerU 返回数据缺少有效 file_urls 或 batch_id")
        raise RuntimeError("[node_pdf_to_md] 未能获取有效上传链接或任务批次 ID")

    file_upload_url = file_upload_urls[0]
    logger.info(f"[node_pdf_to_md] 成功获取上传凭证, 任务批次 ID: {batch_id}")

    # 3. 直传 PDF 文件二进制流至对象存储
    pdf_bytes = pdf_path_boj.read_bytes()
    with requests.Session() as session:
        # 关闭环境变量代理读取, 避免本地代理修改请求头导致对象存储签名失败
        session.trust_env = False
        upload_resp = session.put(file_upload_url, data=pdf_bytes)
        if upload_resp.status_code != 200:
            logger.error(f"[node_pdf_to_md] 上传 PDF 文件失败, HTTP 状态码: {upload_resp.status_code}")
            raise RuntimeError(f"[node_pdf_to_md] 上传 PDF 文件失败, HTTP 状态码: {upload_resp.status_code}")

    logger.info(f"[node_pdf_to_md] PDF 文件上传完成, 准备轮询解析状态: {pdf_path_boj.name}")

    # 4. 轮询等待解析完成
    poll_url = f"{mineru_config.base_url}/extract-results/batch/{batch_id}"
    timeout = 600
    interval = 3 # 轮询探测间隔
    start_time = time.time()

    while True:
        # 4.1 超时判断
        if time.time() - start_time > timeout:
            logger.error(f"[node_pdf_to_md] 轮询 MinerU 结果超时 超过:{timeout}s")
            raise TimeoutError("[node_pdf_to_md] 轮询 MinerU 结果超时")

        # 4.2 发起状态查询并做网络波动容错
        try:
            poll_resp = requests.get(poll_url, headers=headers)
        except Exception as e:
            logger.warning(f"[node_pdf_to_md] 轮询网络请求异常，稍后重试: {e}")
            time.sleep(interval)
            continue

        # 4.3 HTTP 状态码校验
        status_code = poll_resp.status_code
        if status_code != 200:
            if 500 <= status_code < 600:
                logger.warning(f"[node_pdf_to_md] MinerU 服务端偶发临时错误 ({status_code})，继续等待重试...")
                time.sleep(interval)
                continue
            else:
                logger.error(f"[node_pdf_to_md] 轮询状态不可恢复的 HTTP 错误: {status_code}")
                raise RuntimeError(f"[node_pdf_to_md] 轮询状态 HTTP 错误: {status_code}")

        # 4.4 业务状态码校验
        poll_json = poll_resp.json()
        if poll_json.get("code") != 0:
            logger.error(f"[node_pdf_to_md] 轮询业务异常: code={poll_json.get('code')}, msg={poll_json.get('msg')}")
            raise RuntimeError(f"[node_pdf_to_md] 轮询业务异常: {poll_json.get('msg')}")

        # 4.5 解析任务执行状态
        extract_result = poll_json.get("data", {}).get("extract_result", [{}])[0]
        task_state = extract_result.get("state")

        if task_state == "done":
            zip_url = extract_result.get("full_zip_url")
            if not zip_url:
                logger.error("[node_pdf_to_md] 解析状态标记为 done，但 full_zip_url 为空")
                raise RuntimeError("[node_pdf_to_md] 解析结果压缩包下载链接为空")
            logger.info(f"[node_pdf_to_md] 云端解析成功，获取结果压缩包下载地址: {zip_url}")
            return zip_url
        elif task_state == "failed":
            err_msg = extract_result.get("err_msg", "未知错误")
            logger.error(f"[node_pdf_to_md] 云端解析失败: {err_msg}")
            raise RuntimeError(f"[node_pdf_to_md] 云端解析失败: {err_msg}")
        else:
            logger.info(f"[node_pdf_to_md] 云端正在解析中，当前状态: [{task_state}]，持续等待...")
            time.sleep(interval)
            continue

@step_log("step_3_download_and_extract")
def step_3_download_and_extract(zip_url: str, local_dir_obj: Path, stem: str) -> Path:
    """
    步骤三: 下载解析结果压缩包,解压并规整为md文件
    核心职责:
    1. 发起 GET 请求下载 zip_url 压缩包并保存到本地
    2. 创建解压目录并进行防重清理 (shutil.rmtree)
    3. 调用 shutil.unpack_archive 解压 ZIP
    4. 递归检索所有 .md 文件 (Path.rglob)
    5. 按优先级查找目标文档 (同名.md -> full.md -> 第一个.md) 并重命名规整为 {stem}.md
    6. 返回规范化后的本地 Markdown 路径 (Path)
    :param zip_url
    :param local_dir_obj
    :param stem
    :return
    """
    # 1. 发送网络请求下载 MinerU 结果 ZIP 压缩包
    response = requests.get(zip_url, timeout = 60)
    if response.status_code != 200:
        logger.error(f"[node_pdf_to_md] 从 MinerU 下载结果压缩包失败，HTTP 状态码: {response.status_code}")
        raise RuntimeError(f"[node_pdf_to_md] 从 MinerU 下载结果压缩包失败: {response.status_code}")

    zip_file_path = local_dir_obj / f"{stem}_result.zip"
    zip_file_path.write_bytes(response.content)
    logger.info(f"[node_pdf_to_md] 结果压缩包下载完成: {zip_file_path.name}")

    # 2. 准备解压目录 (output/{stem}), 做放重与历史文件清理
    extract_dir = local_dir_obj / stem
    if extract_dir.is_dir():
        logger.warning(f"[node_pdf_to_md] 检测到已存在历史解压目录，先进行清理: {extract_dir}")
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    # 3. 解压 ZIP 压缩包到目标目录
    shutil.unpack_archive(zip_file_path, extract_dir)
    logger.info(f"[node_pdf_to_md] 压缩包解压成功至目录: {extract_dir}")

    # 4. 递归检索所有 Markdown 文件
    md_file_list = list(extract_dir.rglob("*.md"))
    if not md_file_list:
        logger.error(f"[node_pdf_to_md] 解压目录下未检索到任何 .md 文件: {extract_dir}")
        raise RuntimeError(f"[node_pdf_to_md] 解压目录下未检索到任何 .md 文件: {extract_dir}")

    # 5. 按照优先级定位目标 Markdown 文件
    target_md_obj = None

    # 优先级1: 寻找与原 PDF 同名的 Markdown 文件
    for md_obj in md_file_list:
        if md_obj.stem == stem:
            target_md_obj = md_obj
            break

    # 优先级2: 寻找 MinerU 默认全量导出的 full.md
    if not target_md_obj:
        for md_obj in md_file_list:
            if md_obj.name.lower() == "full.md":
                target_md_obj = md_obj
                break

    # 优先级3: 兜底选取检索到的第一个 Markdown 文件
    if not target_md_obj:
        target_md_obj = md_file_list[0]

    # 6. 规整重命名为统一规范文件名 ({stem}.md)
    final_md_path = target_md_obj.rename(target_md_obj.with_name(f"{stem}.md"))
    logger.info(f"[node_pdf_to_md] Markdown 文档处理完成，最终落盘路径: {final_md_path}")

    return final_md_path

@step_log("step_split_upload_and_merge")
def step_split_upload_and_merge(pdf_path_obj: Path, local_dir_obj: Path, total_pages: int, max_pages: int = MAX_PAGES_PER_BATCH) -> Path:
    """
    分卷处理超大 PDF 文件（突破 MinerU 单次 <= 200 页限制）:
    1. 计算分卷区间并使用 PyMuPDF 物理切分生成临时子 PDF
    2. 对每个子 PDF 单独调用 step_2_upload_and_poll 与 step_3_download_and_extract
    3. 按序拼接各个子分卷的 Markdown 内容
    4. 汇总各分卷解析出的图片资源至统一的 images 目录
    5. 输出最终整合后的 Markdown 文件并清理临时分卷文件
    """
    stem = pdf_path_obj.stem
    temp_parts_dir = local_dir_obj / f"{stem}_temp_parts"
    temp_parts_dir.mkdir(parents=True, exist_ok=True)

    final_dir = local_dir_obj / stem
    final_dir.mkdir(parents=True, exist_ok=True)
    final_images_dir = final_dir / "images"
    final_images_dir.mkdir(parents=True, exist_ok=True)

    part_pdf_files: List[Path] = []

    # 1. 切分 PDF
    with fitz.open(pdf_path_obj) as src_doc:
        part_idx = 1
        for start_page in range(0, total_pages, max_pages):
            end_page = min(start_page + max_pages - 1, total_pages - 1)
            part_name = f"{stem}_part{part_idx}.pdf"
            part_path = temp_parts_dir / part_name
            part_pdf_files.append(part_path)

            sub_doc = fitz.open()
            sub_doc.insert_pdf(src_doc, from_page=start_page, to_page=end_page)
            sub_doc.save(part_path)
            sub_doc.close()
            logger.info(f"[node_pdf_to_md] 成功拆分分卷 [{part_idx}]: 页码 {start_page + 1}~{end_page + 1} -> {part_name}")
            part_idx += 1

    # 2. 逐卷上传与解析
    merged_md_sections = []
    total_parts = len(part_pdf_files)
    for idx, part_pdf in enumerate(part_pdf_files, 1):
        logger.info(f"[node_pdf_to_md] 开始处理分卷 [{idx}/{total_parts}]: {part_pdf.name}")
        part_zip_url = step_2_upload_and_poll(part_pdf)
        part_md_path = step_3_download_and_extract(part_zip_url, temp_parts_dir, part_pdf.stem)

        # 读取分卷 Markdown
        part_content = part_md_path.read_text(encoding="utf-8")
        merged_md_sections.append(part_content)

        # 汇总图片资源
        part_img_dir = part_md_path.parent / "images"
        if part_img_dir.is_dir():
            for img_file in part_img_dir.iterdir():
                if img_file.is_file():
                    target_img = final_images_dir / img_file.name
                    try:
                        shutil.copy2(img_file, target_img)
                    except Exception as e:
                        logger.warning(f"[node_pdf_to_md] 复制分卷图片失败: {e}")

    # 3. 规整并保存合并后的 Markdown 文件
    final_md_path = final_dir / f"{stem}.md"
    final_md_content = "\n\n".join(merged_md_sections)
    final_md_path.write_text(final_md_content, encoding="utf-8")
    logger.info(f"[node_pdf_to_md] 超长文档所有分卷合并完成，落盘路径: {final_md_path}")

    # 4. 临时目录清理
    try:
        shutil.rmtree(temp_parts_dir, ignore_errors=True)
    except Exception as e:
        logger.warning(f"[node_pdf_to_md] 清理临时分卷目录异常: {e}")

    return final_md_path

if __name__ == "__main__":
    logger.info("=== 测试 node_pdf_to_md 节点 ===")
    # 构造测试状态
    test_pdf_path = PROJECT_ROOT / "doc" / "万用表RS-12的使用.pdf"
    test_state = create_default_state(
        task_id="test_pdf2md_task_001",
        pdf_path=str(test_pdf_path),
        local_dir=str(PROJECT_ROOT / "output"),
        is_pdf_read_enabled=True
    )
    result_state = node_pdf_to_md(test_state)
    logger.info(f"node_pdf_to_md 执行完成，输出 md_path: {result_state.get('md_path')}")