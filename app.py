import csv
import io
import json
import logging
import os
import urllib.request
import urllib.error
from datetime import date
from functools import wraps
from pathlib import Path

import cv2
import numpy as np
from flask import (
    Flask,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from werkzeug.security import generate_password_hash, check_password_hash

from config import (
    ADMIN_DEFAULT_USERNAME,
    ADMIN_DEFAULT_PASSWORD,
    AI_SYSTEM_PROMPT,
    FACES_DIR,
    LEAVES_DIR,
    MAX_FACE_PHOTOS,
    MYSQL_CONFIG,
    SILICONFLOW_API_KEY,
    SILICONFLOW_API_URL,
    SILICONFLOW_MODEL,
    SIMILARITY_THRESHOLD,
)
from database import Database
from face_engine import FaceEngine
from models import Employee


# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("app")


# Flask 应用初始化

app = Flask(__name__)
app.secret_key = os.urandom(24).hex()

@app.after_request
def no_cache(response):
    """禁止浏览器缓存，确保每次刷新获取最新数据"""
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# ---- 初始化数据库和人脸引擎 ----
try:
    db = Database(MYSQL_CONFIG)
    logger.info("数据库连接成功")
except ConnectionError as e:
    logger.error("数据库连接失败: %s", e)
    raise SystemExit(1) from e

engine = FaceEngine()
logger.info("人脸识别引擎初始化完成")

# 预热：预加载所有已注册人脸的嵌入向量到 GPU 缓存
_emps = db.list_employees()
if _emps:
    _paths = [e.face_path for e in _emps]
    _cached = engine.preload_embeddings(_paths)
    logger.info("嵌入向量缓存预热: %d 个", _cached)

# ---- 创建默认管理员账户（仅首次运行） ----
admin_created = db.create_default_admin(
    ADMIN_DEFAULT_USERNAME,
    generate_password_hash(ADMIN_DEFAULT_PASSWORD),
)
if admin_created:
    logger.info(
        "默认管理员账户已创建: %s / %s", ADMIN_DEFAULT_USERNAME, ADMIN_DEFAULT_PASSWORD
    )



#  登录验证装饰器

def login_required(f):
    """登录验证装饰器 — 未登录则重定向到登录页"""
    @wraps(f)
    def decorated(*args, **kwargs):
        if "admin_id" not in session:
            flash("请先登录", "error")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated



#  共享辅助函数

def _decode_upload(file_storage) -> np.ndarray | None:
    """将上传的文件解码为 OpenCV 图像。失败返回 None。"""
    np_arr = np.frombuffer(file_storage.read(), np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def _find_best_match(
    source_img: np.ndarray, employees: list, source_is_crop: bool = False
) -> tuple:
    """在已注册人员中找到最佳匹配。返回 (best_emp, best_score)。"""
    best_emp, best_score = None, -1.0
    for emp in employees:
        score = engine.compare_files(source_img, emp.face_path, source_is_crop=source_is_crop)
        if score > best_score:
            best_score, best_emp = score, emp
    return best_emp, max(0.0, best_score)


def _get_status_info(row: dict) -> tuple[str, str]:
    """根据考勤行数据派生 (status_text, status_color)。"""
    s = row.get("att_status")
    if s == "成功":
        return "成功", "success"
    if s == "失败":
        return "失败", "danger"
    if s == "请假" or row.get("leave_reason"):
        return "请假", "warning"
    return "未签到", "danger"



#  登录 / 登出

@app.route("/login", methods=["GET", "POST"])
def login():
    """管理员登录"""
    if request.method == "GET":
        return render_template("login.html")

    username = request.form.get("username", "").strip()
    password = request.form.get("password", "").strip()

    if not username or not password:
        flash("请输入用户名和密码", "error")
        return render_template("login.html")

    admin = db.get_admin_by_username(username)
    if not admin or not check_password_hash(admin["password_hash"], password):
        flash("用户名或密码错误", "error")
        return render_template("login.html")

    session["admin_id"] = admin["id"]
    session["admin_username"] = admin["username"]
    flash(f"欢迎回来，{admin['username']}！", "success")
    return redirect(url_for("index"))


@app.route("/logout")
def logout():
    """退出登录"""
    session.clear()
    flash("已退出登录", "success")
    return redirect(url_for("login"))



#  首页
@app.route("/")
@login_required
def index():
    """首页仪表盘（管理员视图）"""
    total, today_ok, today_total, today_leave, pending = db.attendance_stats()
    absentees = db.today_absentees()
    recent = db.recent_attendance(limit=10)
    staff_list = db.list_employees()
    today_ok_list = db.today_ok_records()
    pending_leave_list = db.pending_leaves()
    stats = {
        "total_staff": total,
        "today_success": today_ok,
        "today_total": today_total,
        "today_leave": today_leave,
        "pending_leaves": pending,
        "today_absent": len(absentees),       # 非成功即缺勤
        "expected_checkin": total,             # 应签到 = 全员
    }
    return render_template(
        "index.html",
        stats=stats,
        absentees=absentees,
        recent_records=recent,
        staff_list=staff_list,
        today_ok_list=today_ok_list,
        pending_leave_list=pending_leave_list,
    )



#  人员管理
@app.route("/staff")
@login_required
def staff_list():
    """人员管理页面"""
    employees = db.list_employees()
    return render_template("staff.html", employees=employees)


@app.route("/staff/add", methods=["POST"])
@login_required
def staff_add():
    """添加员工（支持1~3张多角度人脸照片）"""
    name = request.form.get("name", "").strip()
    emp_no = request.form.get("emp_no", "").strip()
    face_files = request.files.getlist("face_photos")

    # 输入校验
    if not name or not emp_no:
        flash("请填写姓名和工号", "error")
        return redirect(url_for("staff_list"))

    valid_files = [f for f in face_files if f.filename != ""]
    if not valid_files:
        flash("请至少上传一张人脸照片", "error")
        return redirect(url_for("staff_list"))

    if len(valid_files) > MAX_FACE_PHOTOS:
        flash(f"最多上传 {MAX_FACE_PHOTOS} 张照片", "error")
        return redirect(url_for("staff_list"))

    if db.get_employee_by_empno(emp_no):
        flash(f"工号 {emp_no} 已存在", "error")
        return redirect(url_for("staff_list"))

    try:
        saved_paths = []

        for idx, face_file in enumerate(valid_files):
            img = _decode_upload(face_file)
            if img is None:
                continue  # 跳过无法读取的图片

            # 检测并裁剪人脸区域
            face = engine.detect_face(img)
            if face is None:
                face = img  # 未检测到人脸则使用整张图

            # 保存原始人脸照片
            suffix = f"_{idx + 1}" if len(valid_files) > 1 else ""
            face_path = str(FACES_DIR / f"{emp_no}{suffix}.png")
            cv2.imwrite(face_path, face)
            saved_paths.append(face_path)

            # 数据增强：从一张照片自动生成多个变体
            augmented = engine.augment_face(face)
            for aug_idx, aug_face in enumerate(augmented):
                aug_path = str(FACES_DIR / f"{emp_no}{suffix}_aug{aug_idx + 1}.png")
                cv2.imwrite(aug_path, aug_face)
                saved_paths.append(aug_path)

        if not saved_paths:
            flash("所有图片都无法读取，请检查文件格式", "error")
            return redirect(url_for("staff_list"))

        # 所有路径用逗号拼接存入 face_path 字段
        combined_path = ",".join(saved_paths)
        db.add_employee(name, emp_no, combined_path)

        # 统计原始和增强样本数
        original_count = len(valid_files)
        aug_count = len(saved_paths) - original_count
        msg = f"人员 {name}（{emp_no}）保存成功 — {original_count} 张原始照片"
        if aug_count > 0:
            msg += f" + {aug_count} 张增强样本"
        flash(msg, "success")
        logger.info("人员保存成功: %s (%s), 总共 %d 张照片（含增强）", name, emp_no, len(saved_paths))
        engine.clear_emb_cache()  # 刷新嵌入缓存

    except Exception as e:
        logger.exception("保存人员失败")
        flash(f"保存失败: {e}", "error")

    return redirect(url_for("staff_list"))


@app.route("/staff/<int:emp_id>/delete", methods=["POST"])
@login_required
def staff_delete(emp_id: int):
    """删除员工"""
    emp = db.get_employee_by_id(emp_id)
    if emp:
        db.delete_employee(emp_id)
        flash(f"人员 {emp.name} 已删除", "success")
        logger.info("人员已删除: %s", emp.name)
        engine.clear_emb_cache()  # 刷新嵌入缓存
    else:
        flash("人员不存在", "error")
    return redirect(url_for("staff_list"))


#  个人签到

@app.route("/checkin")
@login_required
def checkin():
    """个人签到页面"""
    return render_template("checkin.html")


@app.route("/checkin/capture", methods=["POST"])
@login_required
def checkin_capture():
    """接收摄像头截图并执行人脸识别"""
    if "image" not in request.files:
        return jsonify({"success": False, "message": "未收到图片"}), 400

    employees = db.list_employees()
    if not employees:
        return jsonify({"success": False, "message": "请先在人员管理中录入人员信息"}), 400

    try:
        frame = _decode_upload(request.files["image"])
        if frame is None:
            return jsonify({"success": False, "message": "图片解码失败"}), 400

        best_emp, best_score = _find_best_match(frame, employees)

        # 判断是否匹配成功
        if best_emp and best_score >= SIMILARITY_THRESHOLD:
            db.log_attendance(
                best_emp.id, "成功", "个人签到", best_score, "摄像头识别成功"
            )
            logger.info("签到成功: %s (%.2f%%)", best_emp.name, best_score * 100)
            return jsonify({
                "success": True,
                "name": best_emp.name,
                "emp_no": best_emp.emp_no,
                "similarity": round(best_score, 4),
                "message": f"签到成功，相似度 {best_score:.2%}",
            })
        else:
            logger.info("签到失败: 未匹配 (最高 %.2f%%)", max(best_score, 0) * 100)
            return jsonify({
                "success": False,
                "message": f"未匹配到已注册人员，最高相似度 {max(best_score, 0.0):.2%}",
            })

    except Exception as e:
        logger.exception("签到处理失败")
        return jsonify({"success": False, "message": f"处理失败: {e}"}), 500


#  合照签到（批量）

@app.route("/batch")
@login_required
def batch():
    """合照签到页面"""
    return render_template("batch.html")


@app.route("/batch/upload", methods=["POST"])
@login_required
def batch_upload():
    """上传一张合照，检测照片中所有人脸并逐一识别签到"""
    if "photo" not in request.files:
        return jsonify({"error": "请选择一张合照"}), 400

    photo = request.files["photo"]
    if photo.filename == "":
        return jsonify({"error": "请选择一张合照"}), 400

    employees = db.list_employees()
    if not employees:
        return jsonify({"error": "请先录入人员信息"}), 400

    try:
        img = _decode_upload(photo)
        if img is None:
            return jsonify({"error": "图片无法读取，请检查文件格式"}), 400

        # ★ 关键：检测合照中所有人脸（而非只取最大的那个）
        faces = engine.detect_faces_multi(img)
        if not faces:
            return jsonify({
                "total_faces": 0,
                "success_count": 0,
                "results": [],
                "message": "未在合照中检测到任何人脸，请确认照片中有清晰的人脸。",
            })

        results = []
        success_count = 0

        # 对每张检测到的人脸，逐一比对（每人可有多张注册照片）
        for idx, face in enumerate(faces):
            best_emp, best_score = _find_best_match(face, employees, source_is_crop=True)

            if best_emp and best_score >= SIMILARITY_THRESHOLD:
                db.log_attendance(
                    best_emp.id, "成功", "合照签到", best_score, photo.filename
                )
                success_count += 1
                results.append({
                    "face_index": idx + 1,
                    "success": True,
                    "name": best_emp.name,
                    "emp_no": best_emp.emp_no,
                    "similarity": round(best_score, 4),
                })
                logger.info("合照签到成功: 第%d张人脸 → %s (%.2f%%)",
                            idx + 1, best_emp.name, best_score * 100)
            else:
                results.append({
                    "face_index": idx + 1,
                    "success": False,
                    "message": f"未匹配（最高相似度 {max(best_score, 0.0):.2%}）",
                })
                logger.info("合照签到: 第%d张人脸未匹配 (最高 %.2f%%)",
                            idx + 1, max(best_score, 0.0) * 100)

        return jsonify({
            "total_faces": len(faces),
            "success_count": success_count,
            "results": results,
        })

    except Exception as e:
        logger.exception("合照签到处理失败")
        return jsonify({"error": f"处理出错: {e}"}), 500



#  请假审批
@app.route("/leave")
@login_required
def leave_list():
    """请假审批页面"""
    leaves = db.list_leaves()
    return render_template("leave.html", leaves=leaves, today=date.today().isoformat())


@app.route("/leave/submit", methods=["POST"])
@login_required
def leave_submit():
    """提交请假申请"""
    emp_no = request.form.get("emp_no", "").strip()
    leave_date = request.form.get("leave_date", "").strip()
    reason = request.form.get("reason", "").strip()

    if not emp_no or not leave_date or not reason:
        flash("请填写完整的请假信息", "error")
        return redirect(url_for("leave_list"))

    emp = db.get_employee_by_empno(emp_no)
    if not emp:
        flash(f"未找到工号为 {emp_no} 的人员", "error")
        return redirect(url_for("leave_list"))

    # 处理证明文件
    doc_path = None
    doc_file = request.files.get("document")
    if doc_file and doc_file.filename != "":
        dst = LEAVES_DIR / f"{emp_no}_{Path(doc_file.filename).name}"
        doc_file.save(str(dst))
        doc_path = str(dst)

    db.add_leave(emp.id, leave_date, reason, doc_path)
    flash(f"请假申请已提交: {emp.name}（{emp_no}），等待审批", "success")
    logger.info("请假已提交: %s (%s)", emp.name, emp_no)
    return redirect(url_for("leave_list"))


@app.route("/leave/<int:leave_id>/review", methods=["POST"])
@login_required
def leave_review(leave_id: int):
    """审批请假（通过/驳回）"""
    status = request.form.get("status", "").strip()
    if status not in ("通过", "驳回"):
        flash("无效的审批状态", "error")
        return redirect(url_for("leave_list"))

    db.update_leave_status(leave_id, status)
    flash(f"请假已{status}", "success")
    logger.info("请假审批: ID=%d %s", leave_id, status)
    return redirect(url_for("leave_list"))


@app.route("/leave/<int:leave_id>/delete", methods=["POST"])
@login_required
def leave_delete(leave_id: int):
    """删除请假记录"""
    ok = db.delete_leave(leave_id)
    if ok:
        flash("请假记录已删除", "success")
        logger.info("请假记录已删除: ID=%d", leave_id)
    else:
        flash("记录不存在", "error")
    return redirect(url_for("leave_list"))


#  考勤记录

@app.route("/records")
@login_required
def records():
    """考勤记录页面 — 今日全员考勤总览，支持 ?status=成功 过滤"""
    status_filter = request.args.get("status", "").strip()
    if status_filter:
        rows = db.list_attendance(status=status_filter)
    else:
        rows = db.daily_summary()
    return render_template("records.html", records=rows, status_filter=status_filter)


@app.route("/records/export")
@login_required
def records_export():
    """导出考勤记录为 CSV 文件（全员）"""
    rows = db.daily_summary()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "姓名", "工号", "状态", "签到方式", "相似度", "签到时间", "备注",
    ])
    for row in rows:
        has_record = row["att_id"] is not None
        status_text, _ = _get_status_info(row)

        checked_at = row["checked_at"]
        if hasattr(checked_at, "strftime"):
            checked_at = checked_at.strftime("%Y-%m-%d %H:%M")

        sim = f"{row['similarity'] * 100:.1f}%" if has_record else "—"
        mode = row["mode"] or "—"
        notes = row["notes"] or row["leave_reason"] or "—"

        writer.writerow([
            row["name"], row["emp_no"],
            status_text, mode, sim,
            checked_at or "—", notes,
        ])

    output.seek(0)
    buf = io.BytesIO()
    buf.write(output.getvalue().encode("utf-8-sig"))
    buf.seek(0)

    filename = f"attendance_{date.today().isoformat()}.csv"
    return send_file(
        buf,
        mimetype="text/csv",
        as_attachment=True,
        download_name=filename,
    )


@app.route("/records/<int:record_id>/delete", methods=["POST"])
@login_required
def record_delete(record_id: int):
    """删除单条考勤记录"""
    ok = db.delete_attendance(record_id)
    if ok:
        flash("考勤记录已删除", "success")
        logger.info("考勤记录已删除: ID=%d", record_id)
    else:
        flash("记录不存在", "error")
    return redirect(url_for("records"))


@app.route("/records/<int:record_id>/edit", methods=["POST"])
@login_required
def record_edit(record_id: int):
    """修改单条考勤记录的状态或备注（AJAX 或表单提交）"""
    status = request.form.get("status", "").strip()
    notes = request.form.get("notes", "").strip()
    ok = db.update_attendance(record_id, status=status, notes=notes)
    if ok:
        flash("考勤记录已更新", "success")
        logger.info("考勤记录已更新: ID=%d", record_id)
    else:
        flash("记录不存在", "error")

    # AJAX 请求返回 JSON
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return jsonify({"success": ok})
    return redirect(url_for("records"))


@app.route("/records/leave/<int:emp_id>/edit", methods=["POST"])
@login_required
def record_leave_edit(emp_id: int):
    """从考勤页编辑请假备注（AJAX 或表单提交）"""
    notes = request.form.get("notes", "").strip()
    ok = db.update_leave_today(emp_id, notes)
    if ok:
        flash("请假备注已更新", "success")
        logger.info("请假备注已更新: emp_id=%d", emp_id)
    else:
        flash("今日无该员工请假记录", "error")

    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return jsonify({"success": ok})
    return redirect(url_for("records"))


@app.route("/records/row/<int:emp_id>/set-status", methods=["POST"])
@login_required
def record_set_status(emp_id: int):
    """统一设置员工今日状态（成功/请假/未签到）— AJAX"""
    status = request.form.get("status", "").strip()
    if status not in ("成功", "请假", "未签到"):
        return jsonify({"success": False, "message": "无效状态"}), 400

    try:
        db.set_employee_status(emp_id, status)
        flash(f"状态已更新为「{status}」", "success")
        logger.info("手动设置状态: emp_id=%d → %s", emp_id, status)
        return jsonify({"success": True, "status": status})
    except Exception as e:
        logger.exception("设置状态失败")
        return jsonify({"success": False, "message": str(e)}), 500



#  AI 助手聊天（qwen）


@app.route("/chat/ask", methods=["POST"])
@login_required
def chat_ask():
    """AI 助手代理接口 — 硅基流动 SiliconFlow"""
    data = request.get_json(silent=True)
    if not data or "message" not in data:
        return jsonify({"error": "缺少消息内容"}), 400

    user_message = data["message"].strip()
    if not user_message:
        return jsonify({"error": "消息不能为空"}), 400

    if not SILICONFLOW_API_KEY:
        return jsonify({
            "reply": "AI 助手暂未配置。请设置 SILICONFLOW_API_KEY 环境变量。\n"
                    "免费注册: https://siliconflow.cn"
        })

    # ---- 构建实时数据上下文（仪表盘 + 各模块） ----
    total, today_ok, today_total, today_leave, pending = db.attendance_stats()
    absentees = db.today_absentees()
    all_employees = db.list_employees()
    all_leaves = db.list_leaves()
    daily = db.daily_summary()

    ctx_lines = [
        "=== 仪表盘实时数据 ===",
        f"日期：{date.today().isoformat()}",
        f"人员总数：{total}人",
        f"今日签到成功：{today_ok}人 / 应到{total}人",
        f"今日缺勤：{len(absentees)}人",
        f"待审批请假：{pending}条",
        f"今日请假通过：{today_leave}人",
    ]

    # 人员管理模块
    ctx_lines.append("\n=== 人员管理 ===")
    ctx_lines.append(f"已注册人员（{total}人）：")
    for e in all_employees:
        ctx_lines.append(f"  {e.name}(工号{e.emp_no}), 编号ID={e.id}")

    # 考勤记录模块
    ctx_lines.append(f"\n=== 考勤记录（今日 {date.today().isoformat()}）===")
    ctx_lines.append("规则：状态非「成功」都算缺勤。管理员可点击状态徽章或备注文字直接编辑。")
    for r in daily:
        if r["att_id"] is not None:
            status = r["att_status"]
            sim = f"{r['similarity']*100:.1f}%" if r["similarity"] else "—"
            ctx_lines.append(
                f"  {r['name']}({r['emp_no']}) → {status} | {r['mode']} | "
                f"相似度{sim} | {r['checked_at']} | 备注:{r['notes'] or '—'}"
            )
        elif r.get("leave_reason"):
            ctx_lines.append(
                f"  {r['name']}({r['emp_no']}) → 请假 | 原因:{r['leave_reason']}"
            )
        else:
            ctx_lines.append(f"  {r['name']}({r['emp_no']}) → 未签到（缺勤）")

    # 请假审批模块
    ctx_lines.append(f"\n=== 请假审批 ===")
    if all_leaves:
        ctx_lines.append(f"共{len(all_leaves)}条请假记录：")
        for lv in all_leaves:
            ctx_lines.append(
                f"  ID={lv['id']} | {lv['name']}({lv['emp_no']}) | "
                f"{lv['leave_date']} | {lv['reason']} | 状态:{lv['status']}"
            )
    else:
        ctx_lines.append("暂无请假记录")

    context = "\n".join(ctx_lines)
    system_prompt = AI_SYSTEM_PROMPT + "\n\n" + context

    history = data.get("history", [])
    messages = [{"role": "system", "content": system_prompt}]
    for h in history[-20:]:
        messages.append(h)
    messages.append({"role": "user", "content": user_message})

    try:
        req_body = json.dumps({
            "model": SILICONFLOW_MODEL,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1000,
        }).encode("utf-8")

        req = urllib.request.Request(
            SILICONFLOW_API_URL,
            data=req_body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            reply = result["choices"][0]["message"]["content"]
            return jsonify({"reply": reply})

    except urllib.error.HTTPError as e:
        logger.error("SiliconFlow API HTTP error: %s %s", e.code, e.reason)
        return jsonify({"reply": f"AI 服务请求失败（{e.code}），请稍后重试。"})
    except Exception as e:
        logger.exception("SiliconFlow API 调用失败")
        return jsonify({"reply": "AI 服务暂时不可用，请稍后重试。"})


# ============================================================================
#  启动入口
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  人脸考勤系统 - Web 版")
    print("  浏览器访问: http://127.0.0.1:5001")
    print("  按 Ctrl+C 停止服务器")
    print("=" * 60)
    app.run(host="127.0.0.1", port=5001, debug=True, use_reloader=False)
