"""密码加密工具"""

import hashlib

from app.core.config import settings


def encrypt_password(password: str) -> str:
    """
    加密密码（MD5 + 盐值）
    """
    salted_password = password + settings.password_salt
    return hashlib.md5(salted_password.encode()).hexdigest()


def verify_password(plain_password: str, encrypted_password: str) -> bool:
    """验证密码"""
    return encrypt_password(plain_password) == encrypted_password
