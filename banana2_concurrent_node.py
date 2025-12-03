"""
Banana2 Concurrent Node
独立的 Banana2 并发图像生成节点
支持多图输入、并发请求、重试机制和超时控制
"""

import os
import io
import json
import base64
import requests
import time
import uuid
import datetime
import torch
import numpy as np
from PIL import Image
from io import BytesIO
import re
import mimetypes
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Mapping
from typing import Tuple, Optional, List

import oss2


def _log(message):
    """日志输出"""
    print(f"[Banana2-Concurrent] {message}")


def _auto_auth_headers(base_url: str, api_key: str, auth_mode: str):
    """构建认证头"""
    headers = {"Content-Type": "application/json"}
    mode = (auth_mode or "auto").lower()
    if mode == "google_xgoog" or (mode == "auto" and "generativelanguage.googleapis.com" in (base_url or "")):
        headers["x-goog-api-key"] = api_key
    else:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _build_endpoint(base_url: str, model: str, version: str):
    """构建 API 端点 URL"""
    u = (base_url or "").rstrip('/')
    if "/models/" in u and ":generateContent" in u:
        return u

    # Check if base_url already contains a version path
    if u.endswith('/v1') or u.endswith('/v1beta') or u.endswith('/v1alpha'):
        return f"{u}/models/{model}:generateContent"

    ver = (version or "Auto").lower()
    if ver == "auto":
        ver = "v1beta" if "generativelanguage.googleapis.com" in u else "v1"

    return f"{u}/{ver}/models/{model}:generateContent"


def _deep_merge(dst: dict, src: dict):
    """深度合并字典"""
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def _redact_for_log(obj, max_len=256):
    """日志脱敏：隐藏大段 base64 数据"""
    def is_base64_like(s: str) -> bool:
        try:
            return bool(re.fullmatch(r"[A-Za-z0-9+/=\n\r]+", s))
        except Exception:
            return False

    def walk(v):
        if isinstance(v, dict):
            out = {}
            for k, val in v.items():
                if k == "data" and isinstance(val, str) and len(val) > max_len:
                    out[k] = f"[redacted {len(val)} chars]"
                else:
                    out[k] = walk(val)
            return out
        if isinstance(v, list):
            return [walk(x) for x in v]
        if isinstance(v, str):
            if len(v) > max_len and is_base64_like(v):
                return f"[redacted {len(v)} chars]"
            if len(v) > 4096:
                return v[:1024] + f"... [truncated, total {len(v)} chars]"
            return v
        return v

    try:
        return walk(obj)
    except Exception:
        return obj


# 图片下载缓存（避免多个任务重复下载相同图片）
_image_cache = {}
_image_cache_lock = None


class OSSUploadFromData:
    def _build_object_key(self, suggested_name: str, prefix: str) -> str:
        today = datetime.datetime.utcnow()
        date_path = f"{today.year:04d}/{today.month:02d}/{today.day:02d}"
        base = suggested_name.strip() or f"file_{uuid.uuid4().hex[:8]}.bin"
        base = base.replace("\\", "/").split("/")[-1]
        key = "/".join(x.strip("/\\") for x in [prefix, date_path, base] if x)
        return key.replace("\\", "/")

    def _numpy_to_pil(self, arr: np.ndarray) -> Image.Image:
        """
        将单张图像的 numpy 数组安全转换为 PIL，兼容 HWC / CHW，并保留透明度。
        """
        if arr.ndim == 2:
            # 灰度
            return Image.fromarray(arr.astype(np.uint8), mode="L")

        if arr.ndim != 3:
            raise RuntimeError(f"Unsupported image array shape: {arr.shape}")

        h, w, c = arr.shape

        # 如果是 [C, H, W]，则转为 [H, W, C]
        if c not in (1, 3, 4) and h in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
            h, w, c = arr.shape

        arr = arr.astype(np.uint8, copy=False)

        if c == 4:
            return Image.fromarray(arr, mode="RGBA")
        if c == 3:
            return Image.fromarray(arr, mode="RGB")
        if c == 1:
            return Image.fromarray(arr[:, :, 0], mode="L")

        # 非常规通道数，交给 PIL 自行推断
        return Image.fromarray(arr)

    def _img_batch_to_payload(self, image: torch.Tensor) -> List[Tuple[bytes, str, str]]:
        image = image.clamp(0, 1)
        batch = image.shape[0] if len(image.shape) == 4 else 1

        payloads: List[Tuple[bytes, str, str]] = []

        # 单张图
        if batch == 1:
            uid = uuid.uuid4().hex[:8]
            arr = (
                (image[0].cpu().numpy() * 255).astype(np.uint8)
                if len(image.shape) == 4
                else (image.cpu().numpy() * 255).astype(np.uint8)
            )
            pil = self._numpy_to_pil(arr)
            bio = io.BytesIO()
            pil.save(bio, format="PNG")
            payloads.append((bio.getvalue(), f"image_{uid}.png", "image/png"))
            return payloads

        # 多张图 -> 多个 PNG，分别上传
        uid = uuid.uuid4().hex[:8]
        for i in range(batch):
            arr = (image[i].cpu().numpy() * 255).astype(np.uint8)
            pil = self._numpy_to_pil(arr)
            f_bio = io.BytesIO()
            pil.save(f_bio, format="PNG")
            name = f"image_{uid}_{i+1:04d}.png"
            payloads.append((f_bio.getvalue(), name, "image/png"))
        return payloads

    def _audio_input_to_bytes(self, audio: object, file_name: str, mime_type: str) -> Tuple[bytes, str, str]:
        # 0) Already bytes
        if isinstance(audio, (bytes, bytearray)):
            name = file_name.strip() or f"audio_{uuid.uuid4().hex[:8]}.wav"
            mt = mime_type.strip() or (mimetypes.guess_type(name)[0] or "audio/wav")
            return (bytes(audio), name, mt)

        # 1) Try common file path attributes
        potential_path = None
        for attr in ("file", "path", "file_path", "filepath", "audio_path", "filename"):
            if hasattr(audio, attr):
                val = getattr(audio, attr)
                if isinstance(val, str) and os.path.isfile(val):
                    potential_path = val
                    break
        if potential_path is None and isinstance(audio, str) and os.path.isfile(audio):
            potential_path = audio
        if potential_path:
            with open(potential_path, "rb") as f:
                data = f.read()
            name = file_name.strip() or os.path.basename(potential_path)
            mt = mime_type.strip() or (mimetypes.guess_type(name)[0] or "application/octet-stream")
            return data, name, mt

        # 2) Try common export methods to get wav bytes
        for meth in ("to_wav_bytes", "get_wav_bytes"):
            fn = getattr(audio, meth, None)
            if callable(fn):
                try:
                    data = fn()
                    if isinstance(data, (bytes, bytearray)):
                        name = file_name.strip() or f"audio_{uuid.uuid4().hex[:8]}.wav"
                        mt = mime_type.strip() or "audio/wav"
                        return bytes(data), name, mt
                except Exception:
                    pass
        for meth in ("export", "save", "write"):
            fn = getattr(audio, meth, None)
            if callable(fn):
                try:
                    bio = io.BytesIO()
                    try:
                        fn(bio, format="wav")
                    except Exception:
                        fn(bio)
                    data = bio.getvalue()
                    if data:
                        name = file_name.strip() or f"audio_{uuid.uuid4().hex[:8]}.wav"
                        mt = mime_type.strip() or "audio/wav"
                        return data, name, mt
                except Exception:
                    pass

        # 3) Treat as waveform tensor/array
        sr = 44100
        data = None
        if isinstance(audio, Mapping):
            # LazyAudioMap implements Mapping and resolves on first access
            # Fetch sample rate without boolean evaluation on tensors
            for k in ("sample_rate", "sr"):
                try:
                    v = audio.get(k)  # type: ignore[attr-defined]
                    if v is not None:
                        sr = int(v)
                        break
                except Exception:
                    pass
            for k in ("samples", "waveform", "audio"):
                try:
                    v = audio.get(k)  # type: ignore[attr-defined]
                    if v is not None:
                        data = v
                        break
                except Exception:
                    continue
        else:
            data = audio

        # 2.5) Attribute-style containers (e.g., objects with .waveform / .sample_rate)
        if data is audio and not isinstance(audio, (bytes, bytearray)) and not isinstance(audio, Mapping):
            try:
                sr_attr = getattr(audio, "sample_rate", None)
                wf_attr = getattr(audio, "waveform", None)
                if sr_attr is not None and wf_attr is not None:
                    try:
                        sr = int(sr_attr)
                    except Exception:
                        pass
                    data = wf_attr
            except Exception:
                pass

        if isinstance(data, torch.Tensor):
            data_np = data.detach().cpu().numpy()
        else:
            try:
                data_np = np.asarray(data)
            except Exception:
                data_np = None

        if data_np is None or not np.issubdtype(getattr(data_np, "dtype", np.float32), np.number):
            raise RuntimeError(
                "Unsupported AUDIO input: cannot extract waveform or file path from object. "
                "Provide a numeric waveform, a valid file path, or an object with export methods."
            )

        if data_np.ndim == 3 and data_np.shape[0] == 1:
            # [1, C, S] -> [C, S]
            data_np = data_np[0]
        if data_np.ndim == 1:
            data_np = data_np[None, :]
        elif data_np.ndim != 2:
            raise RuntimeError(f"Unsupported audio array shape: {data_np.shape}")

        data_np = data_np.astype(np.float32, copy=False)
        data_np = np.clip(data_np, -1.0, 1.0)
        pcm_i16 = (data_np * 32767.0).astype(np.int16)
        frames = pcm_i16.T.tobytes()

        bio = io.BytesIO()
        with wave.open(bio, "wb") as wf:
            wf.setnchannels(pcm_i16.shape[0])
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(frames)
        name = file_name.strip() or f"audio_{uuid.uuid4().hex[:8]}.wav"
        mt = mime_type.strip() or "audio/wav"
        return bio.getvalue(), name, mt

    def _audio_many_to_payloads(self, audios: object, file_name: str, mime_type: str) -> List[Tuple[bytes, str, str]]:
        """
        支持单个音频对象或多个音频对象（list/tuple 等可迭代）转换为统一的 payload 列表。
        """
        if isinstance(audios, (list, tuple)):
            payloads: List[Tuple[bytes, str, str]] = []
            for a in audios:
                payloads.append(self._audio_input_to_bytes(a, file_name, mime_type))
            return payloads
        # 退化为单个
        return [self._audio_input_to_bytes(audios, file_name, mime_type)]

    def _video_input_to_bytes(self, video: object, file_name: str, mime_type: str) -> Tuple[bytes, str, str]:
        """
        将单个视频对象或路径转换为 payload（三元组）。
        """
        potential_path = None
        for attr in ("file", "path", "file_path", "filepath", "fullpath", "filename"):
            if hasattr(video, attr):
                val = getattr(video, attr)
                if isinstance(val, str) and os.path.isfile(val):
                    potential_path = val
                    break
        if potential_path is None and isinstance(video, str) and os.path.isfile(video):
            potential_path = video
        if potential_path is None:
            raise RuntimeError(
                "Unsupported VIDEO input: cannot resolve file path from object. "
                "Provide a valid file path or object with path attributes."
            )

        with open(potential_path, "rb") as f:
            data = f.read()
        name = file_name.strip() or os.path.basename(potential_path)
        mt = mime_type.strip() or (mimetypes.guess_type(name)[0] or "application/octet-stream")
        return data, name, mt

    def _video_many_to_payloads(self, videos: object, file_name: str, mime_type: str) -> List[Tuple[bytes, str, str]]:
        """
        支持单个视频对象或多个视频对象（list/tuple 等可迭代）转换为 payload 列表。
        """
        if isinstance(videos, (list, tuple)):
            payloads: List[Tuple[bytes, str, str]] = []
            for v in videos:
                payloads.append(self._video_input_to_bytes(v, file_name, mime_type))
            return payloads
        return [self._video_input_to_bytes(videos, file_name, mime_type)]

    def _choose_payloads(
        self,
        image: Optional[torch.Tensor],
        images: Optional[torch.Tensor],
        audio: Optional[object],
        audios: Optional[object],
        video: Optional[object],
        videos: Optional[object],
        file_name: str,
        mime_type: str,
    ) -> List[Tuple[bytes, str, str]]:
        """
        根据优先级选择待上传的载荷。

        优先级：
        1. images（组图）
        2. image（单图）
        3. audios（多音频）
        4. audio（单音频）
        5. videos（多视频）
        6. video（单视频）
        """
        # 1) 图片：优先使用组图端口
        if images is not None:
            return self._img_batch_to_payload(images)
        if image is not None:
            return self._img_batch_to_payload(image)

        # 2) 音频 → WAV
        if audios is not None:
            return self._audio_many_to_payloads(audios, file_name, mime_type)
        if audio is not None:
            return [self._audio_input_to_bytes(audio, file_name, mime_type)]

        # 3) 视频
        if videos is not None:
            return self._video_many_to_payloads(videos, file_name, mime_type)
        if video is not None:
            return [self._video_input_to_bytes(video, file_name, mime_type)]
        # 无有效载荷
        raise RuntimeError("No payload provided. Connect one of: image, audio, or video.")

    def _to_public_url(self, endpoint: str, bucket_name: str, object_key: str) -> str:
        scheme = "https"
        ep = endpoint
        if endpoint.startswith("http://"):
            scheme = "http"
            ep = endpoint[len("http://") :]
        elif endpoint.startswith("https://"):
            ep = endpoint[len("https://") :]
        return f"{scheme}://{bucket_name}.{ep}/{object_key}"

    def upload(
        self,
        endpoint: str,
        access_key_id: str,
        access_key_secret: str,
        bucket_name: str,
        object_prefix: str,
        use_signed_url: bool,
        signed_url_expire_seconds: int,
        image: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        audio: Optional[object] = None,
        audios: Optional[object] = None,
        video: Optional[object] = None,
        videos: Optional[object] = None,
        file_name: str = "",
        mime_type: str = "",
        security_token: str = "",
    ):
        if not endpoint or not access_key_id or not access_key_secret or not bucket_name:
            raise RuntimeError("Missing required OSS configuration.")

        payloads = self._choose_payloads(
            image=image,
            images=images,
            audio=audio,
            audios=audios,
            video=video,
            videos=videos,
            file_name=file_name,
            mime_type=mime_type,
        )

        auth = (
            oss2.StsAuth(access_key_id, access_key_secret, security_token)
            if security_token
            else oss2.Auth(access_key_id, access_key_secret)
        )
        bucket = oss2.Bucket(auth, endpoint, bucket_name)

        urls: List[str] = []

        for payload, suggested_name, content_type in payloads:
            object_key = self._build_object_key(suggested_name, object_prefix)
            headers = {"Content-Type": content_type}
            result = bucket.put_object(object_key, payload, headers=headers)
            if not (200 <= result.status < 300):
                raise RuntimeError(f"Upload failed: status={result.status}")

            url = (
                bucket.sign_url("GET", object_key, signed_url_expire_seconds)
                if use_signed_url
                else self._to_public_url(endpoint, bucket_name, object_key)
            )
            urls.append(url)

        return (urls,)

def _init_cache_lock():
    """初始化缓存锁（延迟导入 threading）"""
    global _image_cache_lock
    if _image_cache_lock is None:
        import threading
        _image_cache_lock = threading.Lock()
    return _image_cache_lock

def _download_image(url: str, proxies=None, timeout=120, use_cache=True):
    """下载图片（带缓存，线程安全）"""
    # 🔧 优化：双重检查锁定模式，避免竞态条件
    if use_cache:
        lock = _init_cache_lock()
        # 第一次检查（不加锁，快速路径）
        if url in _image_cache:
            _log(f"Using cached image: {url[:50]}...")
            return _image_cache[url]
        
        # 第二次检查（加锁，确保线程安全）
        with lock:
            if url in _image_cache:
                _log(f"Using cached image (locked): {url[:50]}...")
                return _image_cache[url]
    
    # 缓存未命中，开始下载
    try:
        _log(f"Downloading image: {url[:50]}...")
        r = requests.get(url, timeout=timeout, proxies=proxies)
        if r.status_code == 200:
            img_data = r.content
            # 缓存图片数据（线程安全）
            if use_cache:
                lock = _init_cache_lock()
                with lock:
                    # 再次检查，避免重复写入（虽然已经下载了）
                    if url not in _image_cache:
                        _image_cache[url] = img_data
                        _log(f"Cached image: {url[:50]}... ({len(img_data)} bytes)")
                    else:
                        _log(f"Image already cached by another thread: {url[:50]}...")
            return img_data
        _log(f"Download failed: HTTP {r.status_code}")
    except Exception as e:
        _log(f"Error downloading image: {e}")
    return None


def _extract_response_images(resp_json, strict_native=False, proxies=None, timeout=120):
    """
    从响应中提取所有可用的图片。
    返回列表，每个元素为字典：{"bytes": b"...", "mime": "image/png", "url": optional str}
    """
    images = []
    seen_urls = set()

    def add_inline(data_str, mime):
        if not data_str:
            return
        try:
            decoded = base64.b64decode(data_str)
            images.append({
                "bytes": decoded,
                "mime": mime or "image/png",
                "url": None,
            })
        except Exception as err:
            _log(f"⚠️ 解码 inline 图像失败: {err}")

    def add_url_resource(url):
        if not url or url in seen_urls:
            return
        seen_urls.add(url)
        img_data = _download_image(url, proxies=proxies, timeout=timeout, use_cache=True)
        if not img_data:
            _log(f"❌ 下载图像失败: {url}")
            return
        mime = mimetypes.guess_type(url)[0] or "image/png"
        images.append({
            "bytes": img_data,
            "mime": mime,
            "url": url,
        })

    # 1) Gemini style: candidates -> parts
    try:
        cands = resp_json.get("candidates") or []
        for cand in cands:
            parts = (cand.get("content") or {}).get("parts") or []
            for p in parts:
                data = p.get("inlineData") or p.get("inline_data") or {}
                mt = (data.get("mimeType") or data.get("mime_type") or "")
                if isinstance(mt, str) and mt.startswith("image/"):
                    add_inline(data.get("data"), mt)

                if not strict_native:
                    text = p.get("text") or ""
                    if text:
                        _log(f"🔍 检查文本中的图像URL: {text[:200]}")
                        for match in re.findall(r'!\[[^\]]*\]\((https?://[^\)]+)\)', text):
                            add_url_resource(match.strip())
                        for match in re.findall(r'(https?://[^\s\)]+\.(?:png|jpg|jpeg|gif|webp|bmp))', text, re.IGNORECASE):
                            add_url_resource(match.strip())
    except Exception as e:
        _log(f"Error in Gemini-style image extraction: {e}")
        import traceback
        _log(traceback.format_exc())

    # 2) OpenAI / DALL·E style data[]
    try:
        data_list = resp_json.get("data")
        if isinstance(data_list, list):
            for item in data_list:
                b64 = item.get("b64_json")
                if b64:
                    add_inline(b64, item.get("mimeType") or "image/png")
                url = item.get("url")
                if url:
                    add_url_resource(url)
    except Exception as e:
        _log(f"Error in OpenAI-style image extraction: {e}")

    # 3) Generic fallbacks (image/images)
    try:
        for k in ["image", "images"]:
            v = resp_json.get(k)
            if isinstance(v, list):
                for item in v:
                    b64 = item.get("base64") or item.get("b64") or item.get("data")
                    if b64:
                        add_inline(b64, item.get("mimeType") or "image/png")
                    url = item.get("url")
                    if url:
                        add_url_resource(url)
    except Exception as e:
        _log(f"Error in fallback image extraction: {e}")

    return images


def _load_config():
    """加载配置文件（仅读取本目录内的配置文件）"""
    try:
        # 只读取当前插件目录内的配置文件
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config.json")
        
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        _log(f"读取配置文件失败: {e}")
    
    # 返回默认空配置
    return {}


def _get_mirror_site_config(mirror_site_name: str):
    """根据镜像站名称获取对应的 url 与 api_key"""
    config = _load_config()
    sites = config.get('mirror_sites', {}) or {}
    if mirror_site_name and mirror_site_name.lower() != 'custom' and mirror_site_name in sites:
        site = sites.get(mirror_site_name, {})
        return {
            'url': site.get('url', ''),
            'api_key': site.get('api_key', '')
        }, config
    return {'url': '', 'api_key': ''}, config


class Banana2ConcurrentNode:
    """Banana2 并发图像生成节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        # 从配置文件读取镜像站选项
        config = _load_config()
        mirror_sites = config.get('mirror_sites', {}) or {}
        mirror_options = list(mirror_sites.keys())
        # 统一包含自定义选项
        mirror_options = ["Custom" if x.lower() == "custom" else x for x in mirror_options]
        if "Custom" not in mirror_options:
            mirror_options.append("Custom")

        # 默认镜像站：优先 nano-banana官方，其次 comfly，再次第一个，最后 Custom
        if "nano-banana官方" in mirror_options:
            default_site = "nano-banana官方"
        elif "comfly" in mirror_options:
            default_site = "comfly"
        elif mirror_options:
            default_site = mirror_options[0]
        else:
            default_site = "Custom"

        return {
            "required": {
                # 提示词文本框
                "prompt": ("STRING", {"default": "生成一张清晰的香水产品图", "multiline": True}),
                # 镜像站选择
                "mirror_site": (mirror_options, {"default": default_site}),
                # API 认证参数
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "base_url": ("STRING", {"default": "https://generativelanguage.googleapis.com"}),

                # 模型选择
                "model": ([
                    "gemini-3-pro-image-preview",
                    "custom"
                ], {"default": "gemini-3-pro-image-preview"}),
                "custom_model": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "当model选择'custom'时，在此输入自定义模型名称"
                }),
                "version": (["Auto", "v1", "v1alpha", "v1beta"], {"default": "Auto"}),
                "auth_mode": (["auto", "google_xgoog", "bearer"], {"default": "auto"}),
                "response_mode": (["TEXT_AND_IMAGE", "IMAGE_ONLY", "TEXT_ONLY"], {"default": "TEXT_AND_IMAGE"}),
                "aspect_ratio": (["Auto","1:1","16:9","9:16","4:3","3:4","3:2","2:3","5:4","4:5","21:9"], {"default": "Auto"}),
                "image_size": (["Auto","1K","2K","4K"], {"default": "Auto"}),

                # 按顺序：temperature -> top_p -> top_k -> max_output_tokens
                "temperature": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 2.0}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0}),
                "top_k": ("INT", {"default": 40, "min": 1, "max": 1000}),
                "max_output_tokens": ("INT", {"default": 2048, "min": 1, "max": 32768}),

                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "strict_native": ("BOOLEAN", {"default": False}),
                "system_instruction": ("STRING", {"default": "", "multiline": True}),
                "image_mime": (["image/png","image/jpeg","image/webp"], {"default": "image/png"}),
                
                # 并发与重试控制
                "concurrency": ("INT", {"default": 3, "min": 1, "max": 100, "tooltip": "同时并发请求数量"}),
                "request_delay": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.1, "tooltip": "并发请求之间的延迟间隔（秒），避免瞬间同时发送"}),
                "retry_times": ("INT", {"default": 1, "min": 1, "max": 10, "tooltip": "单次请求失败后额外重试次数"}),
                "single_timeout": ("INT", {"default": 300, "min": 10, "max": 5000, "tooltip": "单次请求超时时间（秒）"}),
                "total_timeout": ("INT", {"default": 600, "min": 10, "max": 5000, "tooltip": "整个并发+重试过程的总超时时间（秒）"}),
            },
            "optional": {
                # 图片URL文本输入，多行，每行一个URL
                "image_urls_text": ("STRING", {"default": "", "multiline": True, "tooltip": "每行一个图片URL，支持多图"}),
                "extra_payload_json": ("STRING", {"default": "", "multiline": True}),
                # OSS 设置
                "oss_enable_upload": ("BOOLEAN", {"default": False, "tooltip": "勾选后将生成的图片上传到指定 OSS"}),
                "oss_endpoint": ("STRING", {"default": "", "multiline": False}),
                "oss_access_key_id": ("STRING", {"default": "", "password": True}),
                "oss_access_key_secret": ("STRING", {"default": "", "password": True}),
                "oss_bucket_name": ("STRING", {"default": "", "multiline": False}),
                "oss_object_prefix": ("STRING", {"default": "uploads/"}),
                "oss_file_name": ("STRING", {"default": "", "multiline": False, "tooltip": "上传到 OSS 时使用的文件名（留空则自动生成）"}),
                "oss_mime_type": ("STRING", {"default": "", "multiline": False, "tooltip": "上传到 OSS 时使用的 MIME 类型（留空自动使用 image/png）"}),
                "oss_use_signed_url": ("BOOLEAN", {"default": True}),
                "oss_signed_url_expire_seconds": ("INT", {"default": 3600, "min": 60, "max": 604800}),
                "oss_security_token": ("STRING", {"default": "", "password": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("responses", "statuses", "image_urls", "valid_urls", "images")
    FUNCTION = "call_api"
    CATEGORY = "AIYang007_banana"

    def call_api(self, prompt, mirror_site, api_key, base_url, model, custom_model, version, auth_mode,
                 response_mode, aspect_ratio, image_size,
                 temperature, top_p, top_k, max_output_tokens, seed, strict_native,
                 system_instruction, image_mime, concurrency, request_delay, retry_times, single_timeout, total_timeout,
                 image_urls_text="", extra_payload_json="",
                 oss_enable_upload=False, oss_endpoint="", oss_access_key_id="", oss_access_key_secret="",
                 oss_bucket_name="", oss_object_prefix="uploads/", oss_file_name="", oss_mime_type="",
                 oss_use_signed_url=True, oss_signed_url_expire_seconds=3600, oss_security_token=""):
        # 解析镜像站配置与用户输入的优先级
        site_cfg, full_cfg = _get_mirror_site_config(mirror_site)
        global_default_base = full_cfg.get('base_url', 'https://generativelanguage.googleapis.com')

        user_key = (api_key or "").strip()
        user_base = (base_url or "").strip()

        is_custom = (mirror_site or "").lower() == 'custom'
        if is_custom:
            # Custom 必须完全依赖用户输入
            if not user_key or not user_base:
                empty_list_json = json.dumps([], ensure_ascii=False)
                return (
                    json.dumps({"error": "选择 'Custom' 时必须输入 API Key 和 base_url"}, ensure_ascii=False),
                    json.dumps(["error"], ensure_ascii=False),
                    empty_list_json,
                    empty_list_json,
                    torch.zeros(1, 512, 512, 3),
                )
            effective_key = user_key
            effective_base = user_base
        else:
            # 非 Custom：用户输入优先，否则使用配置
            effective_key = user_key if user_key else (site_cfg.get('api_key') or full_cfg.get('api_key') or "").strip()
            effective_base = user_base if user_base else (site_cfg.get('url') or global_default_base)

            if not effective_key:
                empty_list_json = json.dumps([], ensure_ascii=False)
                return (
                    json.dumps({"error": "未提供 API Key，且镜像站配置中也没有可用的Key"}, ensure_ascii=False),
                    json.dumps(["error"], ensure_ascii=False),
                    empty_list_json,
                    empty_list_json,
                    torch.zeros(1, 512, 512, 3),
                )

        _log(f"镜像站: {mirror_site} → 使用 base_url: {effective_base}")
        _log(f"认证模式: {auth_mode}")

        # 🎯 处理自定义模型
        actual_model = model
        if model == "custom":
            if not custom_model.strip():
                empty_list_json = json.dumps([], ensure_ascii=False)
                return (
                    json.dumps({"error": "选择'custom'时必须提供自定义模型名称"}, ensure_ascii=False),
                    json.dumps(["error"], ensure_ascii=False),
                    empty_list_json,
                    empty_list_json,
                    torch.zeros(1, 512, 512, 3),
                )
            actual_model = custom_model.strip()
            _log(f"🔧 使用自定义模型: {actual_model}")

        endpoint = _build_endpoint(effective_base, actual_model, version)
        headers = _auto_auth_headers(effective_base, effective_key, auth_mode)

        # 解析图片URL列表（多行 → 多图；这些图会作为一个任务的多图输入）
        image_urls = []
        if image_urls_text:
            for line in image_urls_text.splitlines():
                url = (line or "").strip()
                if url:
                    image_urls.append(url)

        # Build base parts: 文本 prompt
        base_parts = [{"text": prompt}]

        # Base payload per Gemini docs
        base_payload = {
            "contents": [{"role": "user", "parts": base_parts}],
            "generationConfig": {
                "temperature": float(temperature),
                "topP": float(top_p),
                "topK": int(top_k),
                "maxOutputTokens": int(max_output_tokens),
            },
        }

        # responseModalities
        if response_mode == "IMAGE_ONLY":
            mods = ["IMAGE"]
        elif response_mode == "TEXT_ONLY":
            mods = ["TEXT"]
        else:
            mods = ["TEXT", "IMAGE"]
        base_payload.setdefault("generationConfig", {})["responseModalities"] = mods

        # imageConfig: aspectRatio + imageSize
        gen_cfg = base_payload.setdefault("generationConfig", {})

        if aspect_ratio and aspect_ratio != "Auto":
            gen_cfg.setdefault("imageConfig", {})["aspectRatio"] = aspect_ratio
        if image_size and image_size != "Auto":
            val = str(image_size).upper()
            gen_cfg.setdefault("imageConfig", {})["imageSize"] = val

        # seed (0 means no seed)
        try:
            if isinstance(seed, int) and seed > 0:
                base_payload.setdefault("generationConfig", {})["seed"] = int(seed)
        except Exception:
            pass

        # systemInstruction
        if system_instruction and system_instruction.strip():
            base_payload["systemInstruction"] = {
                "role": "system",
                "parts": [{"text": system_instruction.strip()}]
            }

        # Merge extra JSON
        if extra_payload_json and extra_payload_json.strip():
            try:
                user_extra = json.loads(extra_payload_json)
                base_payload = _deep_merge(base_payload, user_extra)
            except Exception as e:
                _log(f"extra_payload_json parse error: {e}")

        # 不在节点中单独管理代理设置，直接使用系统/requests 默认行为
        proxies = None

        def _call_single(idx, start_time, task_start_delay=0):
            """单个并发任务的执行函数"""
            # 🔧 任务内部延迟：在发送请求前延迟，避免瞬间同时发送
            if task_start_delay > 0:
                _log(f"[Task {idx}] 任务内部延迟 {task_start_delay:.2f}秒后开始执行")
                time.sleep(task_start_delay)
            
            attempts = 0
            last_error = None
            while attempts <= retry_times:
                if time.time() - start_time > total_timeout:
                    return {
                        "index": idx,
                        "status": "timeout_total",
                        "error": f"超过总超时时间 {total_timeout}s",
                        "response": None,
                    }
                attempts += 1
                try:
                    # 为当前任务构建 payload 副本（包含所有输入图片）
                    payload = json.loads(json.dumps(base_payload, ensure_ascii=False))
                    parts_local = [p.copy() for p in base_parts]

                    # 将所有 image_urls 作为多图输入附加到同一个任务中（使用缓存）
                    # 🔧 优化：如果预下载已完成，直接从缓存获取，避免重复检查
                    for url in image_urls:
                        try:
                            # 使用缓存避免重复下载（如果预下载已完成，这里应该直接从缓存获取）
                            img_bytes = _download_image(url, proxies=proxies, timeout=single_timeout, use_cache=True)
                            if img_bytes:
                                b64_img = base64.b64encode(img_bytes).decode()
                                parts_local.append({
                                    "inlineData": {
                                        "mimeType": image_mime or "image/png",
                                        "data": b64_img
                                    }
                                })
                            else:
                                _log(f"[Task {idx}] ⚠️ 图片下载失败或为空: {url}")
                        except Exception as e:
                            _log(f"[Task {idx}] 下载图片失败: {url} -> {e}")
                    payload["contents"][0]["parts"] = parts_local

                    _log(f"[Task {idx}] Request URL: {endpoint}")
                    logged_headers = headers.copy()
                    if "Authorization" in logged_headers:
                        logged_headers["Authorization"] = "Bearer sk-..."
                    if "x-goog-api-key" in logged_headers:
                        logged_headers["x-goog-api-key"] = "AIzaSy..."
                    _log(f"[Task {idx}] Request Headers: {logged_headers}")
                    _log(f"[Task {idx}] Request Payload: {json.dumps(_redact_for_log(payload), ensure_ascii=False, indent=2)}")

                    resp = requests.post(
                        endpoint,
                        headers=headers,
                        data=json.dumps(payload),
                        timeout=single_timeout,
                    )

                    _log(f"[Task {idx}] Response Status Code: {resp.status_code}")
                    if resp.status_code != 200:
                        last_error = f"HTTP {resp.status_code}: {resp.text}"
                        _log(f"[Task {idx}] Error: {last_error}")
                        continue

                    resp_json = resp.json()
                    _log(f"[Task {idx}] Response Body: {json.dumps(_redact_for_log(resp_json), ensure_ascii=False, indent=2)}")
                    status = "success"
                    return {
                        "index": idx,
                        "status": status,
                        "error": None,
                        "response": resp_json,
                    }

                except requests.exceptions.Timeout as e:
                    last_error = f"请求超时 (single_timeout={single_timeout}s): {e}"
                    _log(f"[Task {idx}] {last_error}")
                    continue
                except requests.exceptions.SSLError as e:
                    last_error = f"SSL连接错误: {e}"
                    _log(f"[Task {idx}] {last_error}")
                    continue
                except requests.exceptions.ProxyError as e:
                    last_error = f"代理连接错误: {e}"
                    _log(f"[Task {idx}] {last_error}")
                    continue
                except requests.exceptions.ConnectionError as e:
                    last_error = f"网络连接错误: {e}"
                    _log(f"[Task {idx}] {last_error}")
                    continue
                except Exception as e:
                    last_error = f"请求失败: {e}"
                    _log(f"[Task {idx}] {last_error}")
                    import traceback
                    _log(traceback.format_exc())
                    continue

            return {
                "index": idx,
                "status": "error",
                "error": last_error or "未知错误",
                "response": None,
            }

        # 构建任务列表：一个任务 = 一次完整 Banana 调用（多图输入在同一任务中）
        # 并发数 = 同时跑多少个独立任务
        total_tasks = max(1, int(concurrency) if isinstance(concurrency, (int, float)) else 1)
        tasks = list(range(total_tasks))
        start_time = time.time()

        results = []
        max_workers = max(1, min(total_tasks, len(tasks)))
        
        _log(f"🚀 开始并发执行 {total_tasks} 个任务，请求间隔: {request_delay}秒")
        
        # 🔧 优化：先预下载所有图片（避免每个任务重复下载）
        # 注意：预下载是异步的，不阻塞任务提交
        pre_download_complete = False
        if image_urls:
            _log(f"📥 开始预下载 {len(image_urls)} 张图片（并行下载，使用缓存）...")
            pre_download_start = time.time()

            def download_one(url):
                try:
                    img_data = _download_image(url, proxies=None, timeout=single_timeout, use_cache=True)
                    return img_data is not None
                except Exception as e:
                    _log(f"预下载失败: {url} -> {e}")
                    return False

            # 并行下载所有图片（使用缓存机制避免重复下载）
            with ThreadPoolExecutor(max_workers=min(len(image_urls), 5)) as download_executor:
                download_futures = [download_executor.submit(download_one, url) for url in image_urls]
                success_count = sum(1 for future in download_futures if future.result())

            pre_download_elapsed = time.time() - pre_download_start
            _log(f"✅ 预下载完成，成功: {success_count}/{len(image_urls)} 张（已缓存），耗时: {pre_download_elapsed:.2f}秒")
            pre_download_complete = True
        
        # 🔧 立即启动并发任务（不等待预下载完成，任务内部会使用缓存）
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {}
            task_start_times = {}
            
            # 🔧 立即提交所有任务到线程池（真正并发）
            # 延迟在任务内部执行，避免阻塞主线程
            for idx in tasks:
                task_delay = idx * request_delay if idx > 0 and request_delay > 0 else 0
                task_start_times[idx] = time.time()
                future = executor.submit(_call_single, idx, start_time, task_start_delay=task_delay)
                future_to_idx[future] = idx
            
            submit_end_time = time.time()
            submit_elapsed = submit_end_time - start_time
            _log(f"📊 所有 {total_tasks} 个任务已提交到线程池（耗时: {submit_elapsed:.3f}秒）")
            if request_delay > 0:
                _log(f"   每个任务内部延迟: Task0=0s, Task1={request_delay:.2f}s, Task2={request_delay*2:.2f}s...")
            
            # 等待所有任务完成
            completed_count = 0
            first_complete_time = None
            for future in as_completed(future_to_idx):
                res = future.result()
                results.append(res)
                completed_count += 1
                elapsed = time.time() - start_time
                
                # 记录第一个任务完成时间
                if first_complete_time is None:
                    first_complete_time = time.time()
                    first_task_elapsed = first_complete_time - start_time
                    _log(f"⚡ 第一个任务完成: Task {res.get('index')}，耗时: {first_task_elapsed:.2f}秒")
                
                task_start = task_start_times.get(res.get('index'), start_time)
                task_elapsed = time.time() - task_start
                _log(f"✅ Task {res.get('index')} 完成 ({completed_count}/{total_tasks})，总耗时: {elapsed:.2f}秒，任务执行耗时: {task_elapsed:.2f}秒")
            
            total_elapsed = time.time() - start_time
            _log(f"🎉 所有任务完成！总耗时: {total_elapsed:.2f}秒，平均每个任务: {total_elapsed/total_tasks:.2f}秒")

        # 按 index 排序，保证顺序稳定
        results.sort(key=lambda x: x.get("index", 0))

        responses = [r.get("response") for r in results]
        statuses = [r.get("status") for r in results]

        # 从每个响应中提取所有图像，构建逐任务的图像列表与数据URL
        image_tensors = []
        image_task_index = []  # 用于记录每张图片属于哪个并发任务
        image_urls_output = [[] for _ in range(total_tasks)]
        for r in results:
            idx = r.get("index", 0)
            if idx < 0 or idx >= len(image_urls_output):
                continue
            resp_json = r.get("response")
            if not isinstance(resp_json, dict):
                continue
            try:
                extracted_images = _extract_response_images(
                    resp_json,
                    strict_native=strict_native,
                    timeout=single_timeout,
                )
            except Exception as e:
                _log(f"Error extracting image from response index {idx}: {e}")
                extracted_images = []

            for img_info in extracted_images:
                img_bytes = img_info.get("bytes")
                mime_type = img_info.get("mime") or "image/png"
                if not img_bytes:
                    continue

                try:
                    pil = Image.open(BytesIO(img_bytes))
                    _log(f"Decoded image index={idx} mode={pil.mode} size={pil.size}")
                    pil = pil.convert("RGB")
                except Exception as e:
                    _log(f"PIL open/convert failed for index {idx}: {e}")
                    try:
                        pil = Image.open(BytesIO(img_bytes)).convert("RGB")
                    except Exception as e2:
                        _log(f"PIL retry failed for index {idx}: {e2}")
                        continue

                arr = np.array(pil)
                img_t = torch.from_numpy(arr).float() / 255.0
                if img_t.dim() == 3:
                    img_t = img_t.unsqueeze(0)
                image_tensors.append(img_t)
                image_task_index.append(idx)

                url_str = img_info.get("url")
                if not url_str:
                    try:
                        encoded = base64.b64encode(img_bytes).decode("utf-8")
                        url_str = f"data:{mime_type};base64,{encoded}"
                    except Exception:
                        url_str = None
                if url_str:
                    image_urls_output[idx].append(url_str)

        has_real_images = bool(image_tensors)
        if has_real_images:
            images_out = torch.cat(image_tensors, dim=0)
        else:
            images_out = torch.zeros(1, 512, 512, 3)

        # 调试日志：输出最终 IMAGE 张量的形状与数值范围，方便排查 ComfyUI 显示问题
        try:
            _log(
                f"images_out shape={tuple(images_out.shape)}, "
                f"dtype={getattr(images_out, 'dtype', None)}, "
                f"min={float(images_out.min())}, max={float(images_out.max())}"
            )
        except Exception as _e:
            _log(f"images_out debug log failed: {_e}")

        # 先基于原始 HTTP URL 生成 valid_urls
        valid_urls = [url for group in image_urls_output for url in group]

        # 如果启用 OSS 上传，并且有真实图像，则优先用 OSS URL 覆盖 image_urls_output / valid_urls
        if oss_enable_upload and has_real_images and OSSUploadFromData:
            oss_endpoint = (oss_endpoint or "").strip()
            oss_access_key_id = (oss_access_key_id or "").strip()
            oss_access_key_secret = (oss_access_key_secret or "").strip()
            oss_bucket_name = (oss_bucket_name or "").strip()
            oss_object_prefix = (oss_object_prefix or "uploads/").strip() or "uploads/"
            oss_file_name = (oss_file_name or "").strip()
            oss_mime_type = (oss_mime_type or "").strip()
            oss_security_token = (oss_security_token or "").strip()
            expire_seconds = int(max(60, oss_signed_url_expire_seconds or 3600))

            if not (oss_endpoint and oss_access_key_id and oss_access_key_secret and oss_bucket_name):
                _log("⚠️ OSS上传被跳过：缺少必要配置（endpoint/access_key/bucket）")
            else:
                try:
                    uploader = OSSUploadFromData()
                    upload_tensor = images_out
                    upload_result = uploader.upload(
                        endpoint=oss_endpoint,
                        access_key_id=oss_access_key_id,
                        access_key_secret=oss_access_key_secret,
                        bucket_name=oss_bucket_name,
                        object_prefix=oss_object_prefix,
                        use_signed_url=bool(oss_use_signed_url),
                        signed_url_expire_seconds=expire_seconds,
                        images=upload_tensor,
                        image=None,
                        audio=None,
                        audios=None,
                        video=None,
                        videos=None,
                        file_name=oss_file_name or "",
                        mime_type=oss_mime_type or "",
                        security_token=oss_security_token,
                    )
                    if isinstance(upload_result, tuple) and upload_result:
                        urls = upload_result[0]
                        if isinstance(urls, list) and len(urls) == len(image_task_index):
                            _log(f"✅ OSS上传成功，返回 {len(urls)} 个URL")
                            # 重新构建按任务分组的 image_urls_output
                            image_urls_output = [[] for _ in range(total_tasks)]
                            for img_idx, url in enumerate(urls):
                                t_idx = image_task_index[img_idx]
                                if 0 <= t_idx < len(image_urls_output):
                                    image_urls_output[t_idx].append(url)
                            valid_urls = [url for group in image_urls_output for url in group]
                except Exception as e:
                    _log(f"⚠️ OSS上传失败: {e}")

        # 如果任务成功但没有返回图片，调整状态
        for i, r in enumerate(results):
            idx = r.get("index", i)
            if idx < 0 or idx >= len(image_urls_output):
                continue
            if statuses[i] == "success" and not image_urls_output[idx]:
                statuses[i] = "no_image"

        return (
            json.dumps(responses, ensure_ascii=False),
            json.dumps(statuses, ensure_ascii=False),
            json.dumps(image_urls_output, ensure_ascii=False),
            json.dumps(valid_urls, ensure_ascii=False),
            images_out,
        )


NODE_CLASS_MAPPINGS = {
    "Banana2Concurrent": Banana2ConcurrentNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Banana2Concurrent": "AIYang007_banana2_Concurrent",
}

