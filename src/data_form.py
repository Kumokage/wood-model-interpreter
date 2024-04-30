from __future__ import annotations as _annotations

from typing import Annotated

import fastapi
import fastui.forms
import pydantic
from fastapi import params as fastapi_params
from fastapi import UploadFile
from fastui import AnyComponent
from fastui import components as c
from fastui.forms import FormFile
from pydantic import BaseModel, Field


def patched_fastui_form(model: type[fastui.forms.FormModel]) -> fastapi_params.Depends:
    async def run_fastui_form(request: fastapi.Request):
        async with request.form() as form_data:
            model_data = fastui.forms.unflatten(form_data)

            try:
                yield model.model_validate(model_data)
            except pydantic.ValidationError as e:
                raise fastapi.HTTPException(
                    status_code=422,
                    detail={'form': e.errors(include_input=False, include_url=False, include_context=False)},
                )

    return fastapi.Depends(run_fastui_form)


class ExperimentDataModel(BaseModel):
    experiment_data: Annotated[
        UploadFile, FormFile(accept=".csv", max_size=100_000_000)
    ] = Field(description="CSV file with experimental data")
    parameters_names: str = Field(
        None, description="Parameters to learn from, separated by ;"
    )
    target_name: str = Field(None, description="Name of target parameter")


def data_form() -> list[AnyComponent]:
    return [
        c.Page(
            components=[
                c.Heading(
                    text="Load experiment data", level=1, class_name="text-center mb-5"
                ),
                c.ModelForm(
                    model=ExperimentDataModel,
                    loading=[c.Spinner(text='Model training and interpreter')],
                    display_mode="page",
                    submit_url="/api/load_data",
                ),
            ]
        )
    ]
