from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field, validator


class ROI(BaseModel):
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    name: Optional[str] = None

    @validator("width", "height")
    def positive_dims(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("width/height must be positive")
        return v

    def as_tuple(self):
        return (self.x, self.y, self.width, self.height)


class ROISet(BaseModel):
    video_path: Optional[str] = None
    frame_width: Optional[int] = None
    frame_height: Optional[int] = None
    rois: List[ROI] = Field(default_factory=list)


class AnalysisConfig(BaseModel):
    input_path: str
    output_csv: Optional[str] = None
    stride: int = 1
    use_ffmpeg: bool = False

