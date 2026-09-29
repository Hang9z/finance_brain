import base64
from mimetypes import guess_type
import os
from pathlib import Path
import re
from typing import Dict, List, Tuple
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from minio.deleteobjects import DeleteObject
from common.config.lm_config import lm_config
from common.config.minio_config import minio_config
from common.logging.logger import logger, node_log, step_log
from processor.import_processor.state import ImportGraphState
from utils.clients.minio_utils import get_minio_client
from utils.lm.lm_utils import get_llm_client
from utils.load_prompt import load_prompt
from utils.rate_limit_utils import apply_api_rate_limit
from utils.task_utils import add_running_task, add_done_task


# MinIO 支持的图片格式
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

def is_supported_image(filename: str) -> bool:
    """
    判断文件是否为 MinIO 支持的格式
    :param filename: 图片的文件名或路径名
    :return: True 或 False
    """
    return Path(filename).suffix.lower() in IMAGE_EXTENSIONS

@node_log("node_md_img")
def node_md_img(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 图片处理 (node_md_img)
    为什么叫这个名字: 处理 Markdown 中的图片资源 (Image)。
    未来要实现:
    1. 扫描 Markdown 中的图片链接。
    2. 将图片上传到 MinIO 对象存储。
    3. (可选) 调用多模态模型生成图片描述。
    4. 替换 Markdown 中的图片链接为 MinIO URL。
    """
    # 0. 任务状态追踪: 先把这个节点加入正在运行列表
    if state.get("task_id"):
        add_running_task(state["task_id"], "node_md_img")

    # 1. 从 md文档中提取文本内容, 确定图片存放的目录
    md_content, md_path_obj, md_images_dir_obj = step_1_validate_and_get_data(state)

    # 2. 健壮性: 可能没有图片目录或者目录为空, 跳出整个流程
    if not md_images_dir_obj.is_dir() or len(list(md_images_dir_obj.iterdir())) == 0:
        logger.info(f"[node_md_img] 对应 Markdown 无图片资源目录或无图片文件, 无需处理, 流程结束")
        if state.get("task_id"):
            add_done_task(state["task_id"], "node_md_img")
        return state

    # 3. 扫描图片并截取上下文
    image_info_list: List[Tuple[str, str, Tuple[str, str]]] = step_2_scan_images(md_content, md_images_dir_obj)
    if not image_info_list:
        logger.info(f"[node_md_img] 目录下图片未在正文中被有效引用, 无需替换: {md_path_obj}")
        if state.get("task_id"):
            add_done_task(state["task_id"], "node_md_img")
        return state

    return state

@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state: ImportGraphState) -> Tuple[str, Path, Path]:
    """
    步骤一: 提取md文件中的文本, 重新写入md文件, 找到图片保存的目录
    :param state: 导入全局状态
    :return: (Markdown 文本内容, Markdown文件Path对象, 图片目录Path对象)
    """
    # 1. 获取 md_path 并做校验
    md_path = state.get("md_path")
    if not md_path:
        logger.error("[node_md_img] state 中缺失路径信息, 无法读取 Markdown 文件")
        raise ValueError("[node_md_img] state 中缺失路径信息")

    md_path_obj = Path(md_path)
    if not md_path_obj.is_file():
        logger.error(f"[node_md_img] md_path 指定的文件不存在或不是合法文件: {md_path_obj}")
        raise FileNotFoundError(f"[node_md_img] 文件不存在: {md_path_obj}")

    # 2. 读取 Markdown 全文内容, 回填 state
    md_content = md_path_obj.read_text(encoding="utf-8")
    state["md_content"] = md_content

    # 3. 推导同级图片目录
    md_images_dir_obj = md_path_obj.parent / "images"

    return md_content, md_path_obj, md_images_dir_obj

@step_log("step_2_scan_images")
def step_2_scan_images(md_content: str, md_images_dir_obj: Path) -> List[Tuple[str, str, Tuple[str, str]]]:
    """
    步骤二: 扫描本地图片, 检查是否在正文中被引用, 用正则表达式截取前后 100 字的上下文
    :param md_content: md文档正文文本
    :param md_images_dir_obj: 本地图片所在目录
    :return: 图片信息列表 -> [(图片文件名, 图片绝对路径, (上文, 下文))]
    """
    image_info_list: List[Tuple[str, str, Tuple[str, str]]] = []

    # 遍历图片目录下的所有文件
    for image_file_obj in md_images_dir_obj.iterdir():
        img_name = image_file_obj.name
        image_path = str(image_file_obj.resolve())

        # 过滤不支持的图片类型
        if not is_supported_image(img_name):
            logger.debug(f"[node_md_img] 跳过非支持图片格式文件: {img_name}")
            continue

        # 使用非贪婪正则匹配 Markdown 图片语法: ![...](...图片名...)
        # re.escape 防止文件名中的括号、点等特殊字符导致正则语法错误
        pattern = re.compile(r"!\[.*?]\(.*?" + re.escape(img_name) + r".*?\)")
        search_match = pattern.search(md_content)

        if not search_match:
            logger.warning(f"[node_md_img] 图片 {img_name} 存在于本地目录, 但未在 Markdown 中引用, 跳过")
            continue

        # 截取图片标签前 100 字符与后 100 字符作为语义上下文
        start, end = search_match.start(), search_match.end()
        pre_context = md_content[max(0, start - 100):start]
        post_context = md_content[end:min(end + 100, len(md_content))]

        logger.debug(f"[node_md_img] 成功捕获图片: {img_name}, 位置 [{start}:{end}]")
        image_info_list.append((img_name, image_path, (pre_context, post_context)))

    logger.info(f"[node_md_img] 扫描完成, 共捕获 {len(image_info_list)} 张有效引用图片")
    return image_info_list