

import os
import logging
from datetime import datetime
import pymysql
from pymysql.cursors import DictCursor

from config import (
    MYSQL_CONFIG,
    DATA_DIR,
    FACES_DIR,
    LEAVES_DIR,
    ATTENDANCE_LIST_LIMIT,
)
from models import Employee

logger = logging.getLogger(__name__)


class Database:

    def __init__(self, config: dict | None = None):
        # 确保数据目录存在
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        FACES_DIR.mkdir(parents=True, exist_ok=True)
        LEAVES_DIR.mkdir(parents=True, exist_ok=True)

        # 连接 MySQL
        self._config = config or MYSQL_CONFIG
        self.conn: pymysql.Connection | None = None
        self._connect()
        self._init_schema()
    #  连接管理
    def _connect(self) -> None:
        try:
            self.conn = pymysql.connect(
                host=self._config["host"],
                port=self._config["port"],
                user=self._config["user"],
                password=self._config["password"],
                database=self._config["database"],
                charset=self._config["charset"],
                connect_timeout=self._config.get("connect_timeout", 10),
                cursorclass=DictCursor,
                autocommit=self._config.get("autocommit", False),
            )
            logger.info("数据库连接成功: %s:%s/%s",
                        self._config["host"], self._config["port"], self._config["database"])
        except pymysql.MySQLError as e:
            logger.error("数据库连接失败: %s", e)
            raise ConnectionError(f"无法连接到 MySQL 数据库，请检查配置。\n错误详情: {e}") from e

    def _ensure_connection(self) -> None:
        if self.conn is None:
            self._connect()
            return
        # 距离上次 ping 不足 30 秒则跳过（减少数据库压力）
        now = datetime.now()
        if not hasattr(self, '_last_ping') or (now - self._last_ping).total_seconds() > 30:
            try:
                self.conn.ping(reconnect=True)
                self._last_ping = now
            except (pymysql.MySQLError, AttributeError):
                logger.warning("数据库连接已断开，尝试重连...")
                try:
                    self._connect()
                    self._last_ping = datetime.now()
                    logger.info("数据库重连成功")
                except Exception as e:
                    logger.error("数据库重连失败: %s", e)
                    raise
    #  表结构初始化
    def _init_schema(self) -> None:
        """创建数据库表与性能索引（如果不存在）"""
        self._ensure_connection()
        with self.conn.cursor() as cur:
            # 员工表
            cur.execute("""
                CREATE TABLE IF NOT EXISTS employees (
                    id          INT PRIMARY KEY AUTO_INCREMENT,
                    name        VARCHAR(100)  NOT NULL,
                    emp_no      VARCHAR(50)   NOT NULL UNIQUE,
                    face_path   TEXT           NOT NULL,
                    created_at  DATETIME       NOT NULL
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)

            # 考勤记录表
            cur.execute("""
                CREATE TABLE IF NOT EXISTS attendance (
                    id           INT PRIMARY KEY AUTO_INCREMENT,
                    employee_id  INT          NOT NULL,
                    status       VARCHAR(20)  NOT NULL,
                    mode         VARCHAR(20)  NOT NULL,
                    similarity   FLOAT        NOT NULL,
                    checked_at   DATETIME     NOT NULL,
                    notes        TEXT,
                    FOREIGN KEY (employee_id) REFERENCES employees(id)
                        ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)

            # 请假申请表
            cur.execute("""
                CREATE TABLE IF NOT EXISTS leave_requests (
                    id             INT PRIMARY KEY AUTO_INCREMENT,
                    employee_id    INT          NOT NULL,
                    leave_date     DATE         NOT NULL,
                    reason         TEXT         NOT NULL,
                    status         VARCHAR(20)  NOT NULL DEFAULT '待审批',
                    document_path  TEXT,
                    created_at     DATETIME     NOT NULL,
                    FOREIGN KEY (employee_id) REFERENCES employees(id)
                        ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)

            # 管理员表
            cur.execute("""
                CREATE TABLE IF NOT EXISTS admins (
                    id            INT PRIMARY KEY AUTO_INCREMENT,
                    username      VARCHAR(50)   NOT NULL UNIQUE,
                    password_hash VARCHAR(255)  NOT NULL,
                    created_at    DATETIME      NOT NULL
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)

            # ---- 性能索引（忽略重复创建错误） ----
            _indexes = [
                "CREATE INDEX idx_att_checked ON attendance(checked_at)",
                "CREATE INDEX idx_att_status ON attendance(status)",
                "CREATE INDEX idx_att_emp_date ON attendance(employee_id, checked_at)",
                "CREATE INDEX idx_leave_status ON leave_requests(status)",
            ]
            for sql in _indexes:
                try:
                    cur.execute(sql)
                except pymysql.MySQLError:
                    pass  # 索引已存在

        self.conn.commit()
        logger.info("数据库表结构初始化完成")
    #  员工管理
    def add_employee(self, name: str, emp_no: str, face_path: str) -> None:
        """添加员工"""
        self._ensure_connection()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO employees (name, emp_no, face_path, created_at) "
                "VALUES (%s, %s, %s, %s)",
                (name, emp_no, face_path, now),
            )
        self.conn.commit()

    def delete_employee(self, emp_id: int) -> None:
        self._ensure_connection()

        # 先查出人脸照片路径以便删除文件
        with self.conn.cursor() as cur:
            cur.execute("SELECT face_path FROM employees WHERE id = %s", (emp_id,))
            row = cur.fetchone()

        if row and row["face_path"]:
            # 支持多张照片（逗号分隔的路径）
            for path in row["face_path"].split(","):
                path = path.strip()
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass

        # 删除员工记录（CASCADE 会清理 attendance 和 leave_requests）
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM employees WHERE id = %s", (emp_id,))
        self.conn.commit()

    def list_employees(self) -> list[Employee]:
        """获取所有员工列表（按工号降序）"""
        self._ensure_connection()
        with self.conn.cursor() as cur:
            cur.execute("SELECT * FROM employees ORDER BY emp_no DESC")
            rows = cur.fetchall()
        return [Employee(**r) for r in rows]

    def get_employee_by_empno(self, emp_no: str) -> Employee | None:
        """根据工号查找员工"""
        self._ensure_connection()
        with self.conn.cursor() as cur:
            cur.execute("SELECT * FROM employees WHERE emp_no = %s", (emp_no,))
            row = cur.fetchone()
        return Employee(**row) if row else None

    def get_employee_by_id(self, emp_id: int) -> Employee | None:
        """根据 ID 查找员工"""
        self._ensure_connection()
        with self.conn.cursor() as cur:
            cur.execute("SELECT * FROM employees WHERE id = %s", (emp_id,))
            row = cur.fetchone()
        return Employee(**row) if row else None

    #  管理员账户


    def create_default_admin(self, username: str, password_hash: str) -> bool:
        """创建默认管理员（仅当 admins 表为空时）。返回 True 表示已创建。"""
        self._ensure_connection()
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS cnt FROM admins")
            if cur.fetchone()["cnt"] > 0:
                return False
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cur.execute(
                "INSERT INTO admins (username, password_hash, created_at) VALUES (%s, %s, %s)",
                (username, password_hash, now),
            )
        self.conn.commit()
        return True

    def get_admin_by_username(self, username: str) -> dict | None:
        """根据用户名查找管理员"""
        self._ensure_connection()
        with self.conn.cursor() as cur:
            cur.execute("SELECT * FROM admins WHERE username = %s", (username,))
            row = cur.fetchone()
        return row

    #  考勤签到

    def log_attendance(
            self,
            employee_id: int,
            status: str,
        mode: str,
        similarity: float = 0.0,
        notes: str = "",
    ) -> bool:
        """写入或覆盖今日考勤记录。返回 True 表示新建，False 表示覆盖。"""
        self._ensure_connection()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.conn.cursor() as cur:
            # 查今日是否已有记录（利用 idx_att_emp_date 索引）
            cur.execute(
                "SELECT id FROM attendance "
                "WHERE employee_id = %s "
                "  AND checked_at >= CURDATE() "
                "  AND checked_at < CURDATE() + INTERVAL 1 DAY",
                (employee_id,),
            )
            row = cur.fetchone()
            if row:
                cur.execute(
                    "UPDATE attendance SET status=%s, mode=%s, similarity=%s, "
                    "checked_at=%s, notes=%s WHERE id=%s",
                    (status, mode, float(similarity), now, notes, row["id"]),
                )
            else:
                cur.execute(
                    "INSERT INTO attendance (employee_id, status, mode, similarity, checked_at, notes) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    (employee_id, status, mode, float(similarity), now, notes),
                )
        self.conn.commit()
        return row is None

    def daily_summary(self) -> list[dict]:
        """今日全员考勤总览：所有员工 + 今日签到状态 + 请假信息，按工号升序"""
        self._ensure_connection()
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT
                    e.id, e.name, e.emp_no,
                    a.id AS att_id,
                    a.status AS att_status, a.mode, a.similarity,
                    a.checked_at, IFNULL(a.notes, '') AS notes,
                    l.reason AS leave_reason, l.status AS leave_status
                FROM employees e
                LEFT JOIN attendance a
                    ON e.id = a.employee_id AND DATE(a.checked_at) = CURDATE()
                LEFT JOIN leave_requests l
                    ON e.id = l.employee_id AND l.leave_date = CURDATE()
                ORDER BY e.emp_no ASC
            """)
            return cur.fetchall()

    def list_attendance(self, limit: int = ATTENDANCE_LIST_LIMIT,
                        status: str | None = None) -> list[dict]:
        """获取考勤记录列表（最近 N 条，可选按状态过滤）"""
        self._ensure_connection()
        with self.conn.cursor() as cur:
            if status:
                cur.execute(
                    """
                    SELECT
                        a.id AS att_id,
                        e.name,
                        e.emp_no,
                        a.status AS att_status,
                        a.mode,
                        a.similarity,
                        a.checked_at,
                        IFNULL(a.notes, '') AS notes
                    FROM attendance a
                    JOIN employees e ON e.id = a.employee_id
                    WHERE a.status = %s
                    ORDER BY a.id DESC
                    LIMIT %s
                    """,
                    (status, limit),
                )
            else:
                cur.execute(
                    """
                    SELECT
                        a.id AS att_id,
                        e.name,
                        e.emp_no,
                        a.status AS att_status,
                        a.mode,
                        a.similarity,
                        a.checked_at,
                        IFNULL(a.notes, '') AS notes
                    FROM attendance a
                    JOIN employees e ON e.id = a.employee_id
                    ORDER BY a.id DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
            return cur.fetchall()

    def update_attendance(
        self, record_id: int, status: str = "", notes: str = ""
    ) -> bool:
        """更新单条考勤记录的状态或备注。返回 True 表示成功。"""
        self._ensure_connection()
        with self.conn.cursor() as cur:
            cur.execute("SELECT id FROM attendance WHERE id = %s", (record_id,))
            if not cur.fetchone():
                return False
            parts = []
            args = []
            if status:
                parts.append("status = %s")
                args.append(status)
            if notes:
                parts.append("notes = %s")
                args.append(notes)
            if parts:
                args.append(record_id)
                cur.execute(
                    f"UPDATE attendance SET {', '.join(parts)} WHERE id = %s",
                    args,
                )
        self.conn.commit()
        return True

    def delete_attendance(self, record_id: int) -> bool:
        """删除单条考勤记录。返回 True 表示成功。"""
        self._ensure_connection()
        with self.conn.cursor() as cur:
            cur.execute("SELECT id FROM attendance WHERE id = %s", (record_id,))
            if not cur.fetchone():
                return False
            cur.execute("DELETE FROM attendance WHERE id = %s", (record_id,))
        self.conn.commit()
        return True

    def set_employee_status(self, emp_id: int, status: str) -> bool:
        """
        统一设置员工今日状态（成功/请假/未签到）。

        - 成功：写入 attendance（覆盖旧记录），删除今日请假
        - 请假：写入 leave_requests（待审批），删除今日签到
        - 未签到：删除今日签到和请假，恢复缺勤状态
        """
        self._ensure_connection()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with self.conn.cursor() as cur:
            if status == "成功":
                # 覆盖/新建签到记录，删除请假
                cur.execute(
                    "SELECT id FROM attendance "
                    "WHERE employee_id = %s AND DATE(checked_at) = CURDATE()",
                    (emp_id,),
                )
                row = cur.fetchone()
                if row:
                    cur.execute(
                        "UPDATE attendance SET status='成功', mode='手动修改', "
                        "similarity=0, checked_at=%s, notes='管理员手动标记' WHERE id=%s",
                        (now, row["id"]),
                    )
                else:
                    cur.execute(
                        "INSERT INTO attendance (employee_id, status, mode, similarity, checked_at, notes) "
                        "VALUES (%s, '成功', '手动修改', 0, %s, '管理员手动标记')",
                        (emp_id, now),
                    )
                cur.execute(
                    "DELETE FROM leave_requests WHERE employee_id = %s AND leave_date = CURDATE()",
                    (emp_id,),
                )

            elif status == "请假":
                # 写入请假，删除签到
                cur.execute(
                    "DELETE FROM attendance WHERE employee_id = %s AND DATE(checked_at) = CURDATE()",
                    (emp_id,),
                )
                cur.execute(
                    "SELECT id FROM leave_requests "
                    "WHERE employee_id = %s AND leave_date = CURDATE()",
                    (emp_id,),
                )
                if not cur.fetchone():
                    cur.execute(
                        "INSERT INTO leave_requests (employee_id, leave_date, reason, status, created_at) "
                        "VALUES (%s, CURDATE(), '管理员手动标记', '通过', %s)",
                        (emp_id, now),
                    )

            elif status == "未签到":
                # 清除签到和请假，恢复缺勤
                cur.execute(
                    "DELETE FROM attendance WHERE employee_id = %s AND DATE(checked_at) = CURDATE()",
                    (emp_id,),
                )
                cur.execute(
                    "DELETE FROM leave_requests WHERE employee_id = %s AND leave_date = CURDATE()",
                    (emp_id,),
                )

        self.conn.commit()
        return True

    def attendance_stats(self) -> tuple[int, int, int, int, int]:
        """
        获取今日统计数据（单次查询）。

        Returns:
            (人员总数, 今日成功签到数, 今日签到总数, 今日请假(已通过)数, 待审批请假数)
        """
        self._ensure_connection()
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT
                    (SELECT COUNT(*) FROM employees) AS total_staff,
                    (SELECT COUNT(*) FROM attendance
                     WHERE status = '成功' AND DATE(checked_at) = CURDATE()) AS today_success,
                    (SELECT COUNT(*) FROM attendance
                     WHERE DATE(checked_at) = CURDATE()) AS today_total,
                    (SELECT COUNT(*) FROM leave_requests
                     WHERE status = '通过' AND leave_date = CURDATE()) AS today_leave,
                    (SELECT COUNT(*) FROM leave_requests
                     WHERE status = '待审批') AS pending_leaves
            """)
            row = cur.fetchone()
            return (
                row["total_staff"],
                row["today_success"],
                row["today_total"],
                row["today_leave"],
                row["pending_leaves"],
            )

    def today_absentees(self) -> list[dict]:
        """获取今日缺勤员工（没有成功签到 = 缺勤，包含失败/请假/无记录）"""
        self._ensure_connection()
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT e.id, e.name, e.emp_no,
                       l.reason AS leave_reason,
                       l.status AS leave_status,
                       a.status AS att_status,
                       IFNULL(a.notes, '') AS notes
                FROM employees e
                LEFT JOIN leave_requests l ON e.id = l.employee_id
                    AND l.leave_date = CURDATE()
                LEFT JOIN attendance a ON e.id = a.employee_id
                    AND DATE(a.checked_at) = CURDATE()
                WHERE e.id NOT IN (
                    SELECT employee_id FROM attendance
                    WHERE DATE(checked_at) = CURDATE() AND status = '成功'
                )
                ORDER BY e.emp_no
            """)
            return cur.fetchall()

    def recent_attendance(self, limit: int = 10) -> list[dict]:
        """获取最近 N 条签到记录"""
        self._ensure_connection()
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT a.id, e.name, e.emp_no, a.status, a.mode, a.checked_at
                FROM attendance a
                JOIN employees e ON e.id = a.employee_id
                ORDER BY a.id DESC
                LIMIT %s
                """,
                (limit,),
            )
            return cur.fetchall()

    def today_ok_records(self) -> list[dict]:
        """获取今日成功签到记录（用于仪表盘详情）"""
        self._ensure_connection()
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT e.name, e.emp_no, a.mode, a.similarity, a.checked_at
                FROM attendance a
                JOIN employees e ON e.id = a.employee_id
                WHERE a.status = '成功' AND DATE(a.checked_at) = CURDATE()
                ORDER BY a.checked_at DESC
            """)
            return cur.fetchall()

    def pending_leaves(self) -> list[dict]:
        """获取待审批请假列表（用于仪表盘详情）"""
        self._ensure_connection()
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT l.id, e.name, e.emp_no, l.leave_date, l.reason, l.created_at
                FROM leave_requests l
                JOIN employees e ON e.id = l.employee_id
                WHERE l.status = '待审批'
                ORDER BY l.created_at DESC
            """)
            return cur.fetchall()

    #  请假管理


    def add_leave(
        self,
        employee_id: int,
        leave_date: str,
        reason: str,
        document_path: str | None = None,
    ) -> None:
        """提交请假申请"""
        self._ensure_connection()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO leave_requests "
                "(employee_id, leave_date, reason, status, document_path, created_at) "
                "VALUES (%s, %s, %s, '待审批', %s, %s)",
                (employee_id, leave_date, reason, document_path, now),
            )
        self.conn.commit()

    def list_leaves(self) -> list[dict]:
        """获取请假申请列表"""
        self._ensure_connection()
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    l.id,
                    e.name,
                    e.emp_no,
                    l.leave_date,
                    l.reason,
                    l.status,
                    IFNULL(l.document_path, '') AS document_path,
                    l.created_at
                FROM leave_requests l
                JOIN employees e ON e.id = l.employee_id
                ORDER BY l.id DESC
                """
            )
            return cur.fetchall()

    def update_leave_status(self, leave_id: int, status: str) -> None:
        """审批请假申请（通过/驳回）"""
        self._ensure_connection()
        with self.conn.cursor() as cur:
            cur.execute(
                "UPDATE leave_requests SET status = %s WHERE id = %s",
                (status, leave_id),
            )
        self.conn.commit()

    def delete_leave(self, leave_id: int) -> bool:
        """删除请假记录。返回 True 表示成功。"""
        self._ensure_connection()
        with self.conn.cursor() as cur:
            cur.execute("SELECT id FROM leave_requests WHERE id = %s", (leave_id,))
            if not cur.fetchone():
                return False
            cur.execute("DELETE FROM leave_requests WHERE id = %s", (leave_id,))
        self.conn.commit()
        return True

    def update_leave_today(self, emp_id: int, reason: str) -> bool:
        """更新今日请假原因。返回 True 表示成功。"""
        self._ensure_connection()
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM leave_requests "
                "WHERE employee_id = %s AND leave_date = CURDATE()",
                (emp_id,),
            )
            row = cur.fetchone()
            if not row:
                return False
            cur.execute(
                "UPDATE leave_requests SET reason = %s WHERE id = %s",
                (reason, row["id"]),
            )
        self.conn.commit()
        return True
