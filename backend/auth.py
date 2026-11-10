"""
Authentication & Role-Based Access Control (RBAC) Module
Academic Stress & Clinical Referral Ecosystem
Central University of Karnataka, Kadaganchi, Kalaburagi
Author: Sahil Mina (24UCOMP047) | Guide: Prof. Srikantaiah K C

Provides secure student & counselor authentication:
- PBKDF2HMAC-SHA256 password verification with isolated salts
- Automatic session-scoped AES-256-GCM vault key mounting
- HttpOnly, SameSite cookie session management
- Role-Based Access Control (Student vs. Counselor / Faculty HOD / Admin)
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from fastapi import Request, HTTPException, Response, status

from crypto.db import db_manager
from crypto.vault import EncryptedVault

COOKIE_NAME = "cuk_session_token"


class RegisterRequest(BaseModel):
    student_id: str = Field(..., min_length=3, max_length=30, description="University Enrollment / Employee ID (e.g., 24UCOMP047)")
    full_name: str = Field(..., min_length=2, max_length=100, description="Full Name")
    email: str = Field(..., description="Institutional or Personal Email")
    password: str = Field(..., min_length=6, description="Master Passphrase / Password")
    department: str = Field("Computer Science", description="Academic Department")
    role: str = Field("student", description="Role: student | counselor | hod | admin")


class LoginRequest(BaseModel):
    identifier: str = Field(..., description="Student ID or Email Address")
    password: str = Field(..., description="Account Passphrase")
    role_hint: Optional[str] = Field("student", description="Target portal role hint")


class UserResponse(BaseModel):
    id: int
    student_id: str
    full_name: str
    email: str
    department: str
    role: str


class AuthManager:
    """Handles session authentication and vault mounting for the FastAPI app."""

    def __init__(self, vault: EncryptedVault):
        self.vault = vault

    def register(self, req: RegisterRequest) -> Dict[str, Any]:
        """Registers a new user record and initializes their personal AES-256 vault canary."""
        return db_manager.create_user(
            student_id=req.student_id,
            full_name=req.full_name,
            email=req.email,
            password=req.password,
            department=req.department,
            role=req.role
        )

    def login(
        self,
        req: LoginRequest,
        response: Response,
        client_ip: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Authenticates credentials:
        1. Validates password hash.
        2. Derives user's personal 256-bit AES-GCM vault key from their passphrase.
        3. Mounts the derived key into server-side RAM (session-scoped).
        4. Sets an HttpOnly, SameSite session cookie on the response.
        """
        auth_result = db_manager.verify_credentials(req.identifier, req.password)
        if not auth_result:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid Student ID/Email or password."
            )

        user_dict, vault_key = auth_result

        # Role enforcement for specialized portals
        if req.role_hint in ["counselor", "hod", "admin"] and user_dict["role"] not in ["counselor", "hod", "admin"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied: Account '{user_dict['student_id']}' is not registered with counselor/admin credentials."
            )

        # 1. Mount user's personal AES key into memory
        self.vault.mount_user_key(user_dict["id"], vault_key)

        # 2. Create persistent session token
        session_token = db_manager.create_session(user_dict["id"], client_ip)

        # 3. Issue secure cookie
        response.set_cookie(
            key=COOKIE_NAME,
            value=session_token,
            httponly=True,
            samesite="lax",
            max_age=7 * 24 * 3600,  # 7 days
            secure=False  # Local development over HTTP
        )

        return {
            "status": "SUCCESS",
            "message": f"Welcome back, {user_dict['full_name']}. Vault key mounted in session memory.",
            "user": user_dict,
            "session_token": session_token
        }

    def logout(self, request: Request, response: Response):
        """Wipes user key from memory and deletes session token."""
        token = request.cookies.get(COOKIE_NAME)
        if token:
            user = db_manager.get_user_from_session(token)
            if user:
                self.vault.unmount_user_key(user["id"])
            db_manager.delete_session(token)

        response.delete_cookie(COOKIE_NAME)
        return {"status": "SUCCESS", "message": "Logged out successfully. Vault keys purged from memory."}

    def get_current_user(self, request: Request) -> Optional[Dict[str, Any]]:
        """Resolves authenticated user from session cookie or Authorization header."""
        token = request.cookies.get(COOKIE_NAME)
        if not token:
            # Check Bearer token header fallback
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]

        if not token:
            return None

        return db_manager.get_user_from_session(token)

