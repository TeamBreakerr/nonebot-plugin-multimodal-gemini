import base64
import mimetypes
import os
from pathlib import Path
from typing import Dict, List  # noqa: UP035
from urllib.parse import quote

import aiofiles
import httpx
from nonebot import get_plugin_config, on_command, require
from nonebot.adapters.onebot.v11 import Bot, GroupMessageEvent, Message, MessageEvent
from nonebot.log import logger
from nonebot.params import CommandArg
from nonebot.plugin import PluginMetadata
from nonebot.rule import is_type

from .utils import contains_http_link, crawl_search_keyword, crawl_url_content, remove_all_files_in_dir

require("nonebot_plugin_localstore")
import nonebot_plugin_localstore as store

require("nonebot_plugin_apscheduler")
from nonebot_plugin_apscheduler import scheduler

from .config import Config

# --- 使用新版 SDK ---
from google import genai
from google.genai import types

# 插件元数据
__plugin_meta__ = PluginMetadata(
    name="谷歌 Gemini 多模态助手",
    description="Nonebot2 的谷歌 Gemini 多模态助手，一个命令即可玩转 Gemini 的多模态！",
    usage=(
        "指令：\n"
        "(1) 多模态助手：[引用文件(可选)] + gemini + [问题(可选)]\n"
        "(2) llama搜索：gemini + 搜索[问题]\n\n"
        "支持引用的文件格式有：\n"
        "  音频: .wav, .mp3, .aiff, .aac, .ogg, .flac\n"
        "  图片: .png, .jpeg, .jpg, .webp, .heic, .heif\n"
        "  视频: .mp4, .mpeg, .mov, .avi, .flv, .mpg, .webm, .wmv, .3gpp\n"
        "  文档: .pdf, .js, .py, .txt, .html, .htm, .css, .md, .csv, .xml, .rtf"
    ),
    type="application",
    homepage="https://github.com/zhiyu1998/nonebot-plugin-multimodal-gemini",
    config=Config,
    supported_adapters={"~onebot.v11"},
)

# 加载配置
plugin_config = get_plugin_config(Config)
API_KEY = plugin_config.gm_api_key
MODEL_NAME = plugin_config.gm_model
PROMPT = plugin_config.gm_prompt

# 使用新版 SDK 创建 client（设置 API 版本为 v1alpha）
client = genai.Client(api_key=API_KEY, http_options={'api_version': 'v1alpha'})

# 注册指令
gemini = on_command("gemini", aliases={"Gemini"}, priority=5, rule=is_type(GroupMessageEvent), block=True)


@gemini.handle()
async def chat(bot: Bot, event: MessageEvent, args: Message = CommandArg()):
    query = args.extract_plain_text().strip()
    file_list, text_data = await auto_get_url(bot, event)
    # 如果有引用的文字则加入 query 中
    if text_data:
        query += f"引用：{text_data}"
    if file_list:
        completion: str = await fetch_gemini_req(query, file_list)
        await gemini.finish(Message(completion), reply_message=True)
    # 如果以 "搜索" 开头，则使用专门的搜索功能
    if query.startswith("搜索"):
        completion: str = await gemini_search_extend(query)
        await gemini.finish(Message(completion), reply_message=True)
    # 否则，使用动态检索（基于动态阈值）来决定是否启用 Google 搜索接地
    completion: str = await fetch_gemini_req(query)
    await gemini.finish(Message(completion), reply_message=True)


async def auto_get_url(bot: Bot, event: MessageEvent):
    reply = event.reply
    file_list = []
    text_data = ""
    if reply:
        for segment in reply.message:
            msg_type = segment.type
            msg_data = segment.data
            if msg_type in ["image", "audio", "video"]:
                url = msg_data.get("url") or msg_data.get("file_url")
                file_id = msg_data.get("file") or msg_data.get("file_id")
                local_path = await download_file(url, msg_type, file_id)
                file_data = await to_gemini_init_data(local_path)
                file_list.append(file_data)
            elif msg_type == "file":
                file_id = msg_data.get("file_id")
                file_url_info = await bot.call_api(
                    "get_group_file_url",
                    file_id=file_id,
                    group_id=event.group_id,  # type: ignore
                )
                url = file_url_info["url"]
                local_path = await download_file(url, msg_type, file_id)
                file_data = await to_gemini_init_data(local_path)
                file_list.append(file_data)
            elif msg_type == "forward":
                for forward_segment in msg_data.get("content"):
                    for content_segment in forward_segment.get("message"):
                        msg_type_segment = content_segment.get("type")
                        msg_data_segment = content_segment.get("data")
                        if msg_type_segment == "image":
                            url = msg_data_segment.get("url") or msg_data_segment.get("file_url")
                            file_id = msg_data_segment.get("file") or msg_data_segment.get("file_id")
                            local_path = await download_file(url, msg_type_segment, file_id)
                            file_data = await to_gemini_init_data(local_path)
                            file_list.append(file_data)
                        elif msg_type_segment == "file":
                            file_id = msg_data_segment.get("file_id")
                            file_url_info = await bot.call_api(
                                "get_group_file_url",
                                file_id=file_id,
                                group_id=event.group_id,  # type: ignore
                            )
                            url = file_url_info["url"]
                            local_path = await download_file(url, msg_type_segment, file_id)
                            file_data = await to_gemini_init_data(local_path)
                            file_list.append(file_data)
                        elif msg_type_segment == "text":
                            text_data += f"{msg_data_segment.get('text').strip()}，"
            else:
                text_data = reply.message.extract_plain_text()
    else:
        for segment in event.message:
            if segment.type == "image":
                img_data = segment.data
                file_id = img_data.get("file") or img_data.get("file_id")
                url = img_data.get("url") or img_data.get("file_url")
                local_path = await download_file(url, segment.type, file_id)
                file_data = await to_gemini_init_data(local_path)
                file_list.append(file_data)
    return file_list, text_data


async def gemini_search_extend(query: str) -> str:
    """
    针对以“搜索”开头的请求，使用 Google 搜索工具生成有依据的回答，
    同时支持动态检索阈值配置。
    """
    # 去掉前缀 "搜索"
    query_text = query.replace("搜索", "").strip()

    # 获取配置中的动态检索阈值
    dynamic_threshold = plugin_config.gm_dynamic_threshold
    dynamic_retrieval_config = types.DynamicRetrievalConfig(
        dynamic_threshold=dynamic_threshold
    )

    # 配置 Google 搜索工具
    google_search_tool = types.Tool(
        google_search=types.GoogleSearchRetrieval(
            dynamic_retrieval_config=dynamic_retrieval_config
        )
    )

    response = await client.aio.models.generate_content(
        model=MODEL_NAME,
        contents=[PROMPT, query_text],
        config=types.GenerateContentConfig(
            tools=[google_search_tool]
        )
    )

    grounding = getattr(response, "groundingMetadata", None)
    if grounding and grounding.get("groundingChunks"):
        search_sources = []
        for source in grounding.get("groundingChunks", []):
            web = source.get("web", {})
            search_sources.append(
                f"📌 网站：{web.get('title', '')}\n🌍 来源：{web.get('uri', '')}"
            )
        return response.text + "\n" + "\n".join(search_sources)
    return response.text


async def fetch_gemini_req(query: str, file_list: List[types.Part] = []) -> str:
    old_http_proxy = os.environ.get("HTTP_PROXY")
    old_https_proxy = os.environ.get("HTTPS_PROXY")
    if (old_http_proxy is None or old_https_proxy is None) and plugin_config.gm_proxy:
        os.environ["HTTP_PROXY"] = plugin_config.gm_proxy
        os.environ["HTTPS_PROXY"] = plugin_config.gm_proxy

    # 构造内容列表，文本直接使用字符串，附件使用 types.Part 对象
    contents = [PROMPT, query] if not file_list else [PROMPT, query, *file_list]

    # 使用配置中的动态检索阈值进行动态接地配置
    dynamic_threshold = plugin_config.gm_dynamic_threshold
    dynamic_retrieval_config = types.DynamicRetrievalConfig(
        dynamic_threshold=dynamic_threshold
    )

    response = await client.aio.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
        config=types.GenerateContentConfig(
            tools=[types.Tool(
                google_search=types.GoogleSearchRetrieval(
                    dynamic_retrieval_config=dynamic_retrieval_config
                )
            )]
        )
    )
    if old_http_proxy is None and old_https_proxy is None:
        os.environ.pop("HTTP_PROXY", None)
        os.environ.pop("HTTPS_PROXY", None)
    return response.text


async def to_gemini_init_data(file_path: str):
    """
    读取本地文件后，根据 MIME 类型构造 Gemini 附件数据，
    如果是文本文件则以文本方式传入，否则以二进制方式传入。
    """
    mime_type = mimetypes.guess_type(file_path)[0]
    async with aiofiles.open(file_path, "rb") as f:
        data = await f.read()
        if mime_type and mime_type.startswith("text/"):
            return types.Part.from_text(data.decode("utf-8"))
        else:
            return types.Part.from_bytes(data=data, mime_type=mime_type)


async def download_file(url: str, file_type: str, file_id: str) -> str:
    try:
        local_dir = store.get_plugin_data_file("tmp")
        local_dir.mkdir(parents=True, exist_ok=True)
        if "." in file_id:
            base_name, ext_file_id = file_id.rsplit(".", 1)
            simplified_file_id = base_name[:8]
        async with httpx.AsyncClient() as client_http:
            response = await client_http.get(url)
            response.raise_for_status()
            ext = f".{ext_file_id}"  # type: ignore
            name = "".join(c if c.isalnum() or c in "-_." else "_" for c in Path(simplified_file_id).stem)  # type: ignore
            safe_filename = f"{file_type}_{name}{ext}"
            local_path = local_dir / safe_filename
            async with aiofiles.open(local_path, "wb") as f:
                await f.write(response.content)
        logger.debug(f"文件已成功下载到: {local_path}")
        return str(local_path)
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP 错误：{e.response.status_code} - {e.response.text}")
        raise
    except Exception as e:
        logger.error(f"下载文件时出错：{e}")
        raise


@scheduler.scheduled_job("cron", hour=8, id="job_gemini_clean_tmps")
async def clean_gemini_tmps():
    """
    每日清理 Gemini 临时文件
    """
    local_dir = store.get_plugin_data_file("tmp")
    await remove_all_files_in_dir(local_dir)
