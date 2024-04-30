from __future__ import annotations as _annotations

import polars as pl
from typing import Annotated
from fastapi import APIRouter
from fastui import AnyComponent, FastUI
from fastui import components as c
from fastui.events import GoToEvent

from .data_form import data_form, patched_fastui_form, ExperimentDataModel
from .ml_utils import explain, learn_xgboost

router = APIRouter()


@router.get("/", response_model=FastUI, response_model_exclude_none=True)
def api_index() -> list[AnyComponent]:
    return data_form()


@router.post("/load_data", response_model=FastUI, response_model_exclude_none=True)
async def load_data_form(
    form: Annotated[ExperimentDataModel, patched_fastui_form(ExperimentDataModel)]
):
    df = pl.read_csv(form.experiment_data.file, separator=";")
    X = df[form.parameters_names.split(";")].to_pandas()
    y = df[form.target_name].to_pandas()
    result = learn_xgboost(X, y)
    interpriter_src = explain(result.model, X)
    return [
        c.Heading(text="Learning result", level=2, class_name="text-center"),
        c.Image(
            src=interpriter_src,
            alt="Interpretation of model",
            width=100,
            height=600,
            loading="lazy",
            referrer_policy="no-referrer",
            class_name="border rounded",
        ),
    ]


@router.get("/{path:path}", status_code=404)
async def api_404():
    # so we don't fall through to the index page
    return {"message": "Not Found"}
