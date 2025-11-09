# -*- coding: utf-8 -*-
# Python 3.7
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class WSFrameIn(BaseModel):
    # base64-encoded JPEG/PNG
    type: str = Field(..., regex="^frame$")
    data: str

class WSPredictionOut(BaseModel):
    type: str = "prediction"
    label: str
    score: float
    index: int
    threshold: float
    topk: List[Dict[str, Any]]

class WSInfoOut(BaseModel):
    type: str = "info"
    message: str

class WSErrorOut(BaseModel):
    type: str = "error"
    message: str
