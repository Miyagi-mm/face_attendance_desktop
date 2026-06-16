

from dataclasses import dataclass


@dataclass
class Employee:
    """员工信息"""
    id: int
    name: str
    emp_no: str
    face_path: str
    created_at: str
