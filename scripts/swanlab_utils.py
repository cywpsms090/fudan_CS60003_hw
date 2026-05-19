from __future__ import annotations

import os


def maybe_add_swanlab_callback(
    model,
    *,
    enabled: bool,
    project: str,
    experiment_name: str,
    description: str,
    mode: str = "cloud",
) -> None:
    if not enabled:
        return

    api_key = os.getenv("SWANLAB_API_KEY")
    if not api_key:
        print("SwanLab disabled: SWANLAB_API_KEY is not set.")
        return

    import swanlab
    from swanlab.integration.ultralytics import add_swanlab_callback

    swanlab.login(api_key=api_key)
    add_swanlab_callback(
        model,
        project=project,
        experiment_name=experiment_name,
        description=description,
        mode=mode,
    )
