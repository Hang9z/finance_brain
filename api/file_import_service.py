import os
from pathlib import Path
import shutil
import sys
import uuid
from datetime import datetime
from typing import Any, Dict, List

# 确保直接运行脚本时能正确找到项目根目录
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from starlette.middleware.cors import CORSMiddleware

from common.logging.logger import logger
from processor.import_processor.state import get_default_state
from processor.import_processor.main_graph import kb_import_app
from utils.path_util import PROJECT_ROOT
from utils.task_utils import *

# 1.创建 FastAPI 实例
app = FastAPI(title="导入", description="项目导入服务")

# 2. 配置跨域中间件 (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# 3. 静态页面路由: 返回 page/import.html
@app.get("/import.html", response_class=FileResponse)
async def index_page():
    # 返回前端页面
    html_abs_path = PROJECT_ROOT / "page/import.html"
    logger.info(f"前端页面访问, 文件绝对路径: {html_abs_path}")

    if not os.path.exists(html_abs_path):
        logger.error(f"前端页面文件不存在, 路径: {html_abs_path}")
        raise HTTPException(status_code=404, detail="import.html page not found")

    return FileResponse(
        path=html_abs_path,
        media_type="text/html"
    )

# --------------------------
# 核心接口：多文件上传接口
# 支持多文件批量上传，核心流程：接收文件 → 本地保存 → 启动后台任务
# 访问地址：http://localhost:8000/upload （POST请求，form-data格式传参）
# --------------------------
@app.post("/upload")
async def upload_file(
        background_tasks: BackgroundTasks,
        files: List[UploadFile] = File(...)
):
    """
    文件上传核心接口：
    1. 接收前端上传的文件列表（PDF/MD 为主）
    2. 按「日期/任务ID」分层保存到本地输出目录，避免同名文件冲突
    3. 为每个文件生成全局唯一 TaskID，启动独立的 LangGraph 后台处理任务
    4. 实时更新任务状态，供前端轮询监控进度
    :param background_tasks: FastAPI后台任务对象，用于异步执行LangGraph流程
    :param files: 前端上传的文件列表（form-data格式）
    :return: 包含上传结果和所有任务ID的JSON响应
    """
    # 1. 本地存储根目录: 根目录/output/YYYYMMDD (按日期分层, 便于归档管理)
    today_str = datetime.now().strftime("%Y%m%d")
    date_based_root_dir: Path = PROJECT_ROOT / "output" / today_str

    # 初始化任务ID列表, 用于返回给前端 (一个文件对应一个TaskID)
    task_ids = []

    # 2. 遍历处理每个上传的文件 (支持多文件批量处理)
    for file in files:
        # 生成全局唯一TaskID, 作为该文件全流程跟踪的主键
        task_id = str(uuid.uuid4())
        task_ids.append(task_id)
        logger.info(f"[{task_id}] 开始处理上传文件，文件名：{file.filename}，文件类型：{file.content_type}")

        # 3. 标记 [文件上传] 阶段为 [运行中], 前端第一次轮询时即可感知
        add_running_task(task_id, "upload_file")

        # 4. 构建该任务的本地独立目录: 在根目录基础上/TaskID, 避免多任务同名文件冲突
        task_local_dir: Path = date_based_root_dir / task_id
        task_local_dir.mkdir(parents=True, exist_ok=True)

        # 5. 构建上传文件的本地保存绝对路径
        local_file_abs_path: Path = task_local_dir / file.filename

        # 6. 将上传的文件二进制流保存到本地临时目录
        with local_file_abs_path.open("wb") as file_buffer:
            shutil.copyfileobj(file.file, file_buffer)
        logger.info(f"[{task_id}] 开始处理上传文件, 文件名: {file.filename}, 文件类型: {file.content_type}")

        # 7. 标记 [文件上传] 阶段为 [已完成]
        add_done_task(task_id, "upload_file")

        # 8. 将 LangGraph 全流程处理加入 FastAPI 后台任务队列 (非阻塞立刻返回)
        background_tasks.add_task(
            run_graph_task,
            task_id,
            str(task_local_dir),
            str(local_file_abs_path)
        )
        logger.info(f"[{task_id}] 已将LangGraph全流程加入后台任务，任务已启动")

    # 9. 遍历完毕，向前端立即响应成功信息与 TaskID 列表
    logger.info(f"多文件上传处理完毕，共处理{len(files)}个文件，生成TaskID列表：{task_ids}")
    return {
        "code": 200,
        "message": f"Files uploaded successfully, total: {len(files)}",
        "task_ids": task_ids
    }

# --------------------------
# 核心接口：任务状态查询接口
# 前端轮询此接口获取单个任务的处理进度和状态
# 访问地址：http://localhost:8000/status/{task_id} （GET请求）
# --------------------------
@app.get("/status/{task_id}", summary="任务状态查询", description="根据TaskID查询单个文件的处理进度和全局状态")
async def get_task_progress(task_id: str):
    """
    任务状态查询接口：
    前端轮询此接口（如每2秒1次），获取任务的实时处理进度。
    数据直接读取内存态字典（task_utils.py 维护），毫秒级极速响应，无磁盘/数据库 IO 开销。
    :param task_id: 全局唯一任务ID（由/upload接口返回）
    :return: 包含任务全局状态、已完成节点中文列表、运行中节点列表的JSON响应
    """
    # 构造任务状态返回体
    task_status_info: Dict[str, Any] = {
        "code": 200,
        "task_id": task_id,
        "status": get_task_status(task_id), # 全局状态: pending / processing / completed / failed
        "done_list": get_done_task_list(task_id), # 已完成节点中文列表
        "running_list": get_running_task_list(task_id) # 正在运行的节点列表
    }

    # 记录状态查询日志
    logger.info(
        f"[{task_id}] 任务状态查询, 当前状态: {task_status_info['status']}, 已完成节点: {task_status_info['done_list']}",
    )
    return task_status_info

# 执行 Lang graph 全流程接口
def run_graph_task(task_id: str, local_dir: str, local_file_path: str):
    """
    LangGraph全流程执行后台任务
    核心流程：初始化状态 → 流式执行图节点 → 实时更新任务状态 → 异常捕获
    任务状态更新：pending → processing → completed/failed
    节点进度更新：每完成一个节点，将节点名加入 done_list，供前端轮询查看
    :param task_id: 全局唯一任务ID，关联单个文件的全流程处理
    :param local_dir: 该任务的本地文件存储目录（含临时文件/解析结果）
    :param local_file_path: 上传文件的本地绝对路径
    """
    try:
        # 1. 更新任务全局状态为: 处理中 (processing)
        update_task_status(task_id, "processing")
        logger.info(f"[{task_id}] 开始执行LangGraph全流程，本地文件路径：{local_file_path}")

        # 2. 初始化 LangGraph 状态: 加载默认状态 + 注入当前任务的核心参数
        init_state = get_default_state()
        init_state["task_id"] = task_id # 任务 ID 关联
        init_state["local_dir"] = local_dir # 任务本地目录
        init_state["local_file_path"] = local_file_path # 上传文件的本地路径

        # 3. 流式执行 LangGraph 全流程 (使用 .stream 模式实时监听每个节点的产出)
        # event就是一个节点，每次一个节点执行完 stream 会返回 {"节点名": 节点产出}
        # {
        #     "node_entry": {
        #         "task_id": "c7a8b9e0-1234-5678-abcd",
        #         "file_title": "万用表RS-12的使用",
        #         "is_pdf_read_enabled": True,
        #         "is_md_read_enabled": False,
        #         "pdf_path": "output/20261011/task_id/万用表RS-12的使用.pdf",
        #         "md_path": None,
        #         # ...其他字段
        #     }
        # }
        for event in kb_import_app.stream(init_state):
            for node_name, node_result in event.items():
                logger.info(f"[{task_id}] LangGraph节点执行完成：{node_name}")
                # 关键：将完成的节点名加入【已完成列表】，前端轮询 /status 时即可看到进度推进
                add_done_task(task_id, node_name)

        # 4. 全图 7 个节点全部执行完成, 更新任务状态为: 已完成 (completed)
        update_task_status(task_id, "completed")
        logger.info(f"[{task_id}] LangGraph全流程执行完毕，任务完成")

    except Exception as e:
        # 5. 捕获全流程未处理异常，更新任务状态为：失败 (failed)，并记录堆栈日志
        update_task_status(task_id, "failed")
        logger.error(f"[{task_id}] LangGraph全流程执行失败，异常信息：{str(e)}", exc_info=True)

# --------------------------
# 服务启动入口
# --------------------------
if __name__ == "__main__":
    """服务启动入口：本地开发环境直接运行"""
    logger.info("File Import Service 服务启动中...")
    # 启动 uvicorn 服务，监听本地 8000 端口
    uvicorn.run(
        app="api.file_import_service:app",  # 对应文件名与实例名
        host="127.0.0.1",
        port=8000,
        reload=False
    )