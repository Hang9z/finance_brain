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

    # 4. 调用 VLM 视觉大模型生成图片中文语义摘要
    summary_image_dict: Dict[str, str] = step_3_image_summary(image_info_list, md_path_obj.stem)

    # 5. 上传图片至 MinIO 对象存储并获取网络URL
    image_url_dict: Dict[str, str] = step_4_upload_images_get_url(image_info_list, md_path_obj.stem)

    # 6. 替换正文中的本地图片引用为 Alt描述 + 网络 URL
    md_content: str = step_5_md_content_image_replace(md_content, summary_image_dict, image_url_dict)

    # 7. 将替换后的 Markdown 磁盘化备份
    md_path_obj: Path = step_6_backup_new_md_content(md_content, md_path_obj)

    # 8. 更新 State 状态数据
    state["md_content"] = md_content
    state["md_path"] = str(md_path_obj)

    # 9. 更新任务状态
    if state.get("task_id"):
        add_done_task(state["task_id"], "node_md_img")

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

@step_log("step_3_image_summary")
def step_3_image_summary(
        image_info_list: List[Tuple[str, str, Tuple[str, str]]],
        root_folder: str
) -> Dict[str, str]:
    """
    步骤三: 调用 VLM 结合上下文生成摘要
    :param image_info_list: 图片信息列表 [(文件名, 完整路径, (上文, 下文))]
    :param root_folder: 文档所属根标识(如文件名)，供提示词渲染使用
    :return: 字典 -> {图片文件名: 图片摘要描述}
    """
    summary_image_dict: Dict[str, str] = {}

    # 1. 获取 VLM 客户端
    vl_model = get_llm_client(model=lm_config.vl_model)
    chain = vl_model | StrOutputParser()

    # 2. 逐张处理图片
    for img_name, img_path, img_context in image_info_list:
        try:
            # 渲染提示词
            prompt_text = load_prompt("image_summary", root_folder=root_folder, image_content=img_context)

            # 读取图片二进制并进行 Base64 编码
            img_path_obj = Path(img_path)
            img_base64_str = base64.b64encode(img_path_obj.read_bytes()).decode("utf-8")
            mime_type = guess_type(img_name)[0] or "image/jpeg"

            # 组装 LangChain 多模态 HumanMessage 结构
            message = HumanMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{img_base64_str}"}
                    },
                    {
                        "type": "text",
                        "text": prompt_text
                    }
                ]
            )

            # 调用 API 速率限制 (滑动窗口限速, 防止触发第三方接口风控)
            apply_api_rate_limit()

            # 调用模型生成摘要
            summary = chain.invoke([message]).strip()
            summary_image_dict[img_name] = summary
            logger.info(f"[node_md_img] ({len(summary_image_dict)}/{len(image_info_list)}) 图片 {img_name} 摘要生成完成: {summary}")


        except Exception as e:
            logger.warning(f"[node_md_img] 图片 {img_name} 生成摘要失败: {e}，将使用文件名作为兜底描述")
            summary_image_dict[img_name] = Path(img_name).stem

    logger.info(f"[node_md_img] 视觉摘要处理完毕，成功生成 {len(summary_image_dict)} 条描述")

    return summary_image_dict

@step_log("step_4_upload_images_get_url")
def step_4_upload_images_get_url(
        image_info_list: List[Tuple[str, str, Tuple[str, str]]],
        stem: str
) -> Dict[str, str]:
    """
    步骤四: 上传图片到 MinIO 对象存储, 实现幂等覆盖并获取访问 HTTP URL
    :param image_info_list: 图片信息列表 [(文件名, 完整路径, (上文, 下文))]
    :param stem: 知识库文件根名，作为 MinIO 目录隔离前缀
    :return: 字典 -> {图片文件名: HTTP URL}
    """
    minio_client = get_minio_client()
    image_url_dict: Dict[str, str] = {}

    # 规范化对象存储路径, 去掉多余前导斜杠 (MinIO 前缀不能以 / 开头)
    base_img_dir = minio_config.minio_img_dir.strip("/")
    target_prefix = f"{base_img_dir}/{stem}"

    # 1. 幂等性保障: 先批量删除 MinIO 中该文档历史遗留的所有图片
    try:
        existing_objects = minio_client.list_objects(
            bucket_name=minio_config.bucket_name,
            prefix=f"{target_prefix}/",
            recursive=True
        )
        delete_list = [DeleteObject(obj.object_name) for obj in existing_objects]
        if delete_list:
            errors = minio_client.remove_objects(minio_config.bucket_name, delete_list)
            for err in errors:
                logger.warning(f"[node_md_img] 清理 MinIO 旧文件异常: {err}")
            logger.info(f"[node_md_img] 已成功清理 MinIO 历史旧图片共 {len(delete_list)} 个")
    except Exception as e:
        logger.warning(f"[node_md_img] MinIO 旧图片清理流程出现异常: {e}")

    # 2. 逐张上传新图片
    protocol = "https" if minio_config.minio_secure else "http"
    for img_name, img_path, _ in image_info_list:
        object_name = f"{target_prefix}/{img_name}"
        content_type = guess_type(img_name)[0] or "application/octet-stream"

        try:
            minio_client.fput_object(
                bucket_name=minio_config.bucket_name,
                object_name=object_name,
                file_path=img_path,
                content_type=content_type
            )
            # 拼接标准访问 URL
            url = f"{protocol}://{minio_config.endpoint}/{minio_config.bucket_name}/{object_name}"
            image_url_dict[img_name] = url
            logger.info(f"[node_md_img] 图片已成功上传至 MinIO: {img_name} -> {url}")
        except Exception as e:
            logger.error(f"[node_md_img] 上传图片 {img_name} 至 MinIO 失败: {e}", exc_info=True)

    logger.info(f"[node_md_img] MinIO 图片上传完成, 共成功上传 {len(image_url_dict)} 张")
    return image_url_dict

@step_log("step_5_md_content_image_replace")
def step_5_md_content_image_replace(
        old_md_content: str,
        summary_image_dict: Dict[str, str],
        image_url_dict: Dict[str, str]
) -> str:
    """
    步骤五: 将 Markdown 原文中本地图片链接替换为: ![图片摘要描述](MinIO网络URL)
    :param old_md_content: 原始 Markdown 内容
    :param summary_image_dict: 图片摘要映射字典 {图片文件名: 摘要描述}
    :param image_url_dict: 图片网络 URL 映射字典 {图片文件名: MinIO URL}
    :return: 替换完成的全新 Markdown 内容
    """
    new_md_content = old_md_content

    for img_name, img_summary in summary_image_dict.items():
        img_url = image_url_dict.get(img_name)
        if not img_url:
            logger.warning(f"[node_md_img] 图片 {img_name} 缺少上传后的网络 URL，跳过该图片替换")
            continue

        # 构造精确匹配该图片的 Markdown 标签正则
        pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(img_name) + r".*?\)")

        # 使用 lambda 回调避免反斜杠和转义字符被当成正则分组引用
        new_md_content = pattern.sub(lambda _: f"![{img_summary}]({img_url})", new_md_content)
        logger.debug(f"[node_md_img] 已成功替换图片: {img_name} -> ![{img_summary}]({img_url})")

    logger.info("[node_md_img] Markdown 正文所有图片已完成语义与网络 URL 替换")
    return new_md_content

@step_log("step_6_backup_new_md_content")
def step_6_backup_new_md_content(md_content_new: str, md_path_obj: Path) -> Path:
    """
    步骤六: 将替换后的全新 Markdown 内容物理备份至磁盘 (命名为: xxx_new.md)
    :param md_content_new: 替换后的 Markdown 文本
    :param md_path_obj: 原始 Markdown 的 Path 对象
    :return: 备份生成的新文件 Path 对象
    """
    # 构造同级下的备份文件路径, 如: 万用表RS-12的使用_new.md
    md_path_obj_new = md_path_obj.with_name(f"{md_path_obj.stem}_new.md")

    # 写入新内容
    md_path_obj_new.write_text(md_content_new, encoding="utf-8")
    logger.info(f"[node_md_img] 已成功将增强后的 Markdown 备份落盘至: {md_path_obj_new}")

    return md_path_obj_new


if __name__ == "__main__":
    from utils.path_util import PROJECT_ROOT

    logger.info("===== 开始执行 node_md_img 单节点单元测试 =====")

    # 指向本地已存在的 MinerU 解析输出测试文档
    test_md_path = PROJECT_ROOT / "output" / "万用表RS-12的使用" / "万用表RS-12的使用.md"

    if not test_md_path.exists():
        logger.error(f"[node_md_img 测试] 测试文件不存在: {test_md_path}，请先确保前置节点已成功解析 PDF")
    else:
        # 构造模拟状态
        test_state: ImportGraphState = {
            "task_id": "test_task_img_001",
            "md_path": str(test_md_path),
            "md_content": "",
            "local_dir": str(test_md_path.parent),
            "local_file_path": str(test_md_path),
            "file_title": test_md_path.stem,
            "is_md_read_enabled": True,
            "is_pdf_read_enabled": False,
            "chunks": [],
            "item_name": "",
            "embeddings_content": [],
            "pdf_path": "",
        }

        # 执行图片增强与上传处理
        result_state = node_md_img(test_state)

        logger.info("===== node_md_img 单节点单元测试完成 =====")
        logger.info(f"新 Markdown 路径 (md_path): {result_state.get('md_path')}")
        logger.info(f"替换后文档长度 (md_content 字符数): {len(result_state.get('md_content', ''))}")

        # 打印正文中前 3 个图片标签看看效果
        replaced_tags = re.findall(r"!\[.*?\]\(.*?\)", result_state.get("md_content", ""))
        logger.info(f"正文共包含图片标签 {len(replaced_tags)} 个，前 3 个样例:")
        for idx, tag in enumerate(replaced_tags[:3], start=1):
            logger.info(f"  样例 {idx}: {tag}")