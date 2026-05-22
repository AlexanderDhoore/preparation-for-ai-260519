from __future__ import annotations

from pathlib import Path

script = Path("/root/preparation-for-ai/08-practical-mlops/lab1/track-image-model.py")
source = script.read_text(encoding="utf-8")
source = source.replace(
    """SELECTED_LABELS = [
    "TODO_FILL_LABEL_1",
    "TODO_FILL_LABEL_2",
    "TODO_FILL_LABEL_3",
    "TODO_FILL_LABEL_4",
    "TODO_FILL_LABEL_5",
]""",
    """SELECTED_LABELS = [
    "pizza",
    "ramen",
    "greek_salad",
    "hamburger",
    "sushi",
]""",
)
source = source.replace(
    'RUN_LABEL = "TODO_FILL_RUN_LABEL"',
    'RUN_LABEL = "frozen-backbone"',
)
source = source.replace(
    "predictions = TODO_CALL_THE_MODEL",
    "predictions = model(images)",
)

namespace = {"__name__": "__main__", "__file__": str(script)}
exec(compile(source, str(script), "exec"), namespace)
