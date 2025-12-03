## Banana2 并发节点（AIYang007_banana2_Concurrent）

独立的 **Banana2 并发图像生成节点插件**，支持多图输入、并发请求、重试机制、超时控制，并内置阿里云 OSS 上传能力（在同一个节点中完成生成和上传）。

## 功能特性

- ✅ **多图输入支持**：通过文本格式输入多个图片 URL（每行一个），所有图片会作为一个任务的多图输入  
- ✅ **并发请求**：支持同时发起多个独立任务，每个任务都包含完整的多图输入  
- ✅ **并发间隔延迟**：可配置请求之间的延迟间隔，避免瞬间同时发送导致服务端限流  
- ✅ **重试机制**：单次请求失败后可自动重试  
- ✅ **超时控制**：支持单次请求超时和总超时时间设置  
- ✅ **OSS 自动上传**：可选将生成的图片自动上传到阿里云 OSS，并直接返回 OSS URL（`image_urls` / `valid_urls`）  
- ✅ **单文件实现**：OSS 上传逻辑已合并进 `banana2_concurrent_node.py`，无需额外的 `oss_node.py` 文件  

## 安装

将本目录（`AIYang_banana`）复制到 ComfyUI 的 `custom_nodes` 目录下即可。  

## 使用方法

### 基本参数

- **prompt**：提示词文本  
- **mirror_site**：镜像站选择（从 `config.json` 中读取，可配置多个 Banana/Gemini 代理）  
- **api_key / base_url**：API Key 与基础 URL，支持直接填写或从镜像站配置中读取  
- **model / custom_model**：模型名称，支持选择固定模型或自定义模型名  
- **version**：API 版本（Auto / v1 / v1alpha / v1beta）  
- **auth_mode**：认证模式（auto / google_xgoog / bearer）  
- **response_mode**：返回类型（TEXT_AND_IMAGE / IMAGE_ONLY / TEXT_ONLY）  
- **aspect_ratio / image_size**：图片宽高比与尺寸参数  
- **temperature / top_p / top_k / max_output_tokens / seed**：采样与生成参数  
- **strict_native**：是否严格仅使用原生图片通道（关闭则会尝试从文本中解析图片 URL）  
- **system_instruction**：系统提示词（作为 system role 传入）  
- **image_mime**：上传/内联图片的 MIME 类型（image/png / image/jpeg / image/webp）  
- **concurrency**：并发数（1–100，默认 3）  
- **request_delay**：任务之间的延迟（秒，默认 0.5）  
- **retry_times**：单任务失败重试次数（默认 1）  
- **single_timeout**：单次请求超时（秒，默认 300）  
- **total_timeout**：整个并发任务总超时（秒，默认 600）  
- **image_urls_text**：多行图片 URL（每行一个），会作为多图输入附加到请求里  
- **extra_payload_json**：附加到请求 body 的自定义 JSON 片段（将与自动生成的 `generationConfig` 深度合并）  

### 输出

- **responses**：每个任务的完整响应 JSON（数组格式）  
- **statuses**：每个任务的状态（数组格式：`success` / `error` / `no_image` / `timeout_total`），与并发数量一一对应  
- **image_urls**：二维数组，`image_urls[i]` 对应第 `i` 个并发任务的全部图片 URL：  
  - 未启用 OSS 上传时：  
    - 如果 Banana 原始返回的是 HTTP 图片 URL，则这里是原始 HTTP URL  
    - 如果 Banana 仅返回 base64 图像，则会转成 `data:image/...;base64,...` 的 data URL  
  - 启用 OSS 上传且上传成功时：  
    - 这里会被 **OSS 返回的图片 URL 覆盖**（与任务一一对应）  
- **valid_urls**：按照任务顺序展开的一维图片 URL 列表（逻辑与 `image_urls` 相同，只是拍平成一维数组）  
- **images**：ComfyUI IMAGE 类型，包含所有返回的图像（Banana 返回的图片会被解码为 `(N, H, W, 3)` 的浮点张量）  

## 并发逻辑说明

1. **多图输入**：`image_urls_text` 中的多个 URL 会被打包成一个任务的多图输入。  
2. **并发执行**：`concurrency` 参数决定同时执行多少个独立任务。  
3. **延迟间隔**：每个任务会延迟 `idx * request_delay` 秒后提交，避免瞬间同时发送。  

例如：  
- `concurrency = 3`，`request_delay = 0.5`  
- Task 0：立即提交  
- Task 1：延迟 0.5 秒后提交  
- Task 2：延迟 1.0 秒后提交  

这样可以减少服务端限流风险，同时保持并发效率。

## 配置文件

插件目录内包含 `config.json` 配置文件，可以配置：  
- **默认 API Key**  
- **默认 Base URL**  
- **镜像站列表（mirror_sites）**（每个镜像站可以配置自己的 `url` 与 `api_key`）  

如果不需要配置文件，可以直接在节点中填写 `api_key` 和 `base_url`，或选择 `Custom` 镜像站并手动输入全部参数。

## OSS 上传（可选）

在节点底部启用 `oss_enable_upload`，可将生成的图片自动上传到阿里云 OSS，并将 OSS URL 回填到：  
- **`image_urls` / `valid_urls` 输出中**（不再额外单独提供 `urls` 输出插口）。  

**必填：**

- `oss_endpoint`  
- `oss_access_key_id`  
- `oss_access_key_secret`  
- `oss_bucket_name`  

**可选：**

- `oss_object_prefix`  
- `oss_file_name`  
- `oss_mime_type`  
- `oss_use_signed_url`  
- `oss_signed_url_expire_seconds`  
- `oss_security_token`  

内部使用的是合并进本文件的 `OSSUploadFromData` 工具类，会将 ComfyUI IMAGE 张量转换为 PNG 并逐张上传，返回 OSS URL 列表。

## 注意事项

- **并发数较高（如 5+）时**，服务端可能会限流或返回错误，建议适当增大 `request_delay`。  
- **启用 OSS 上传** 会增加一次网络请求和带宽占用，请根据实际需求开启。  
- 插件完全独立，所有逻辑（Banana 调用 + 图片解码 + OSS 上传）都在 `banana2_concurrent_node.py` 中实现。  


