from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class WhatsAppNumberRequest(BaseModel):
    whatsapp_number: Optional[str] = ""

class WhatsAppMcpRequest(BaseModel):
    whatsapp_mcp_number: Optional[str] = ""

class WhatsAppNotificationNumberRequest(BaseModel):
    whatsapp_notifications_number: Optional[str] = ""

class WhatsAppNotificationRequest(BaseModel):
    enabled: bool

class ProfileUpdateRequest(BaseModel):
    onboardingAnswers: Dict[str, Any]
    personalInfo: Dict[str, Any]
    preferences: Dict[str, Any]

class CompleteProfileRequest(BaseModel):
    needs_pa: str = Field(..., alias="needs-pa")  # "yes" or "no"
    whatsapp_notifications_number: str

    class Config:
        allow_population_by_field_name = True
