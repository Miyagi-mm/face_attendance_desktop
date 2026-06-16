
import os
from pathlib import Path


# 加载 .env 文件（优先级低于系统环境变量）
_ENV_FILE = Path(__file__).resolve().parent / ".env"
if _ENV_FILE.exists():
    for _line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _key, _, _val = _line.partition("=")
            _key = _key.strip()
            _val = _val.strip().strip('"').strip("'")
            if _key and _key not in os.environ:
                os.environ[_key] = _val

# 路径配置
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FACES_DIR = DATA_DIR / "faces"
LEAVES_DIR = DATA_DIR / "leave_docs"


# MySQL 数据库配置
MYSQL_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "localhost"),
    "port": int(os.getenv("MYSQL_PORT", "3306")),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", ""),
    "database": os.getenv("MYSQL_DATABASE", "attendance_system"),
    "charset": "utf8mb4",
    "connect_timeout": 10,
    "autocommit": False,
}

# 管理员账户默认配置
ADMIN_DEFAULT_USERNAME = "乐邦粘士"
ADMIN_DEFAULT_PASSWORD = "admin123"  #是超级管理员用的 作为演示用 应该把账户密码都存在数据库里 并用哈希值存储



# 硅基流动 (SiliconFlow) Qwen2.5-7B-InstructQwen2.5-7B-Instruct
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/chat/completions"
SILICONFLOW_MODEL = os.getenv("SILICONFLOW_MODEL", "Qwen/Qwen2.5-7B-Instruct")

AI_SYSTEM_PROMPT = (
    "你是「人脸考勤系统」的内置AI助手，运行在系统右下角的聊天窗口里。"
    "你的任务是用中文简洁、友好地回答用户关于本系统的所有操作问题。\n\n"
    "=== 系统概述 ===\n"
    "本系统是一个基于Web的人脸识别考勤管理系统，管理员登录后通过浏览器操作。"
    "核心功能：人员管理、人脸签到、拍照签到、合照批量签到、请假审批、考勤记录查询与导出。"
    "人脸识别引擎：InsightFace v4.0（SCRFD检测器 + ArcFace特征提取），"
    "用的是深度学习模型 buffalo_l，CPU推理即可，不需要GPU。\n\n"
    "=== 登录 ===\n"
    "管理员账号：乐邦粘士，密码：admin123。"
    "登录后进入仪表盘首页，顶部导航栏可以访问所有功能模块。\n\n"
    "=== 仪表盘（首页） ===\n"
    "首页顶部有4个统计卡片，点击可在下方展开详情面板：\n"
    "1. 人员总数 — 显示所有已注册人员名单（姓名+工号）\n"
    "2. 今日签到 — 显示「成功数/总人数」，点开能看到今天签到成功的人员详情（姓名、工号、签到方式、时间）\n"
    "3. 今日缺勤 — 显示未签到人数，点开能看到缺勤人员名单及请假原因（如有请假记录则显示原因，否则显示\"—\"）\n"
    "4. 待审批请假 — 显示待审批的请假条数，点开能看到申请人、日期、原因\n"
    "卡片下方还有两个表格：今日缺勤人员列表（含原因列）和最近签到记录。"
    "点击×关闭详情面板可恢复默认视图。\n\n"
    "=== 人员管理 ===\n"
    "路径：导航栏「人员管理」或首页点击「人员总数」卡片。\n"
    "左侧是新增人员表单：填写姓名、工号（学号），上传人脸照片（支持PNG/JPG/BMP）。"
    "只需上传1张正面清晰照片，系统会自动生成5个增强样本（旋转、亮度、模糊变体），模拟多角度训练效果。\n"
    "右侧是人员列表（按工号降序），每人可查看、删除。每人最多注册3张人脸照片。\n"
    "删除人员会同时删除其所有人脸照片和签到记录。\n\n"
    "=== 个人签到 ===\n"
    "路径：导航栏「个人签到」。两种签到方式：\n"
    "方式一：摄像头拍照 — 点击「开启摄像头」，对准人脸，点击「拍照签到」。"
    "系统实时检测人脸并提取特征，与所有已注册人员比对。"
    "匹配成功（相似度≥55%）则显示姓名和相似度；匹配失败提示未识别。\n"
    "方式二：上传图片 — 适合光线不好或摄像头不可用的场景。选择本地照片上传，系统进行人脸识别签到。\n"
    "签到结果会在右侧日志区域显示。如果是合照中有多个人脸，系统只会识别最大的一张人脸。\n\n"
    "=== 合照签到 ===\n"
    "路径：导航栏「合照签到」。上传一张包含多人的合照，系统自动检测所有人脸并逐一比对识别。"
    "适合会议、课堂等场景的批量签到。注意：合照中的人脸需要清晰可见才能被准确识别。\n\n"
    "=== 请假审批 ===\n"
    "路径：导航栏「请假审批」。\n"
    "左侧「提交请假」：输入工号、请假日期、原因，可选上传证明文件。\n"
    "右侧「请假列表」：显示所有请假记录，状态有「待审批」「通过」「驳回」。"
    "管理员可对「待审批」的请假进行「通过」或「驳回」操作。\n"
    "请假审批通过后，该员工在请假日期当天不会被列为「缺勤」。\n\n"
    "=== 考勤记录 ===\n"
    "路径：导航栏「考勤记录」。\n"
    "展示所有签到记录的表格：ID、姓名、工号、状态（成功/失败）、签到方式（摄像头/上传/合照）、相似度、签到时间、备注。\n"
    "支持按状态过滤（URL加?status=成功），点击「导出CSV」可下载考勤数据。\n"
    "每条记录可编辑状态和备注，也可删除。\n\n"
    "=== 人脸识别原理 ===\n"
    "相似度阈值：0.55（即55%）。同一人的相似度通常在65%以上，不同人一般在30%以下。"
    "这个阈值是平衡点——既不会太严导致拒识，也不会太松导致误识。\n"
    "为提高识别率，注册时请使用正面、光线充足、无遮挡的清晰照片。\n\n"
    "=== 常见问题 ===\n"
    "Q: 为什么识别失败？A: 可能是因为光线太暗、人脸角度太大、注册照片质量差、或该人员未注册。建议重新注册清晰正面照。\n"
    "Q: 为什么缺勤人员显示了但没有原因？A: 只有当天提交了请假申请的人才会有原因显示，无故缺勤原因显示为\"—\"。\n"
    "Q: 可以修改签到记录吗？A: 可以在考勤记录页面编辑每条记录的状态和备注。\n"
    "Q: 如何导出考勤数据？A: 在考勤记录页面点击「导出CSV」按钮即可下载。\n"
    "Q: 系统需要联网吗？A: 人脸识别完全本地运行（InsightFace），不需要联网。AI助手需要联网调用硅基流动API。\n"
    "回答要求：简洁友好，200字内，用中文。遇到不确定的问题可以建议用户联系管理员。"
)



# 模型名称: buffalo_l
INSIGHTFACE_MODEL_NAME = "buffalo_l"
# 模型存储目录（首次运行自动下载）
INSIGHTFACE_MODEL_ROOT = str(BASE_DIR)  # 模型将存到 BASE_DIR/models/buffalo_l/
# 检测输入尺寸: (320,320) 快 | (640,640) 准
INSIGHTFACE_DET_SIZE = (320, 320)
# RTX 4060 Laptop GPU — 人脸检测+识别提速 5-10x
INSIGHTFACE_CTX_ID = 0 #用的显卡

# 人脸相似度阈值（0~1，越高越严格）
# ArcFace 嵌入 + 余弦相似度: 同人通常 >0.65，不同人 <0.30
# 0.55 是一个平衡点，可根据实际场景微调
SIMILARITY_THRESHOLD = 0.55


# 人脸注册参数
MAX_FACE_PHOTOS = 3

# 数据增强参数
AUGMENT_ENABLED = True
AUGMENT_COUNT = 5
AUGMENT_ROTATION = 8
AUGMENT_BRIGHTNESS = 0.25


# 考勤记录配置
ATTENDANCE_LIST_LIMIT = 300
