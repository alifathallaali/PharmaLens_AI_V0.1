from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey, Text
from sqlalchemy.orm import relationship
from enterprise.database import Base

class Company(Base):
    __tablename__ = "companies"
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    industry = Column(String(100), default="Pharmaceutical")
    country = Column(String(100), default="Egypt")
    created_at = Column(DateTime, default=datetime.utcnow)
    users = relationship("User", back_populates="company")
    workspaces = relationship("Workspace", back_populates="company")

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(255))
    role = Column(String(50), default="analyst")
    is_active = Column(Boolean, default=True)
    company_id = Column(Integer, ForeignKey("companies.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    company = relationship("Company", back_populates="users")

class Workspace(Base):
    __tablename__ = "workspaces"
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    company = relationship("Company", back_populates="workspaces")

class Dataset(Base):
    __tablename__ = "datasets"
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    file_name = Column(String(255))
    file_path = Column(String(1000))
    workspace_id = Column(Integer, ForeignKey("workspaces.id"))
    uploaded_by = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)

class Subscription(Base):
    __tablename__ = "subscriptions"
    id = Column(Integer, primary_key=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False)
    plan = Column(String(50), default="free")
    status = Column(String(50), default="active")
    max_users = Column(Integer, default=1)
    max_datasets = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.utcnow)

class AuditLog(Base):
    __tablename__ = "audit_logs"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    action = Column(String(255))
    details = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
