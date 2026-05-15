import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

from pathlib import Path
from typing import Any

import gradio as gr

from lensieve.agent.photo_agent import PhotoAgent
from lensieve.image import load_image
from lensieve.retrieval.schema import SearchResult

DISPLAY_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}


def to_display_image(path: Path, cache_dir: Path) -> str | None:
    if path.suffix.lower() in DISPLAY_EXTENSIONS:
        return str(path)

    cache_dir.mkdir(parents=True, exist_ok=True)

    out_path = cache_dir / f"{path.stem}.jpg"

    try:
        img = load_image(path)
        img.save(out_path, "JPEG", quality=90)
        return str(out_path)
    except Exception:
        return None


def build_app(agent: PhotoAgent, cache_dir: Path) -> gr.Blocks:
    def run_query(user_query: str) -> tuple[str, Any]:
        user_query = user_query.strip()

        if not user_query:
            return "Enter a query.", gr.update(value=[])  # , visible=False)

        try:
            result = agent.run_once(user_query)

            if isinstance(result, str):
                text = result
                image_paths: list[str] = []

            elif isinstance(result, SearchResult):
                hits = result.hits
                num_hits = len(hits)
                text = "No matches found." if num_hits == 0 else f"Showing top {num_hits} matches."
                image_paths = [
                    display_path
                    for hit in result.hits
                    if (display_path := to_display_image(Path(hit.path), cache_dir)) is not None
                ]
            else:
                text = f"Internal error: result type is {type(result).__name__}"
                image_paths = []

            return text, gr.update(value=image_paths)  # , visible=bool(image_paths))

        except Exception as e:
            return f"Error: {e}", gr.update(value=[])  # , visible=False)

    with gr.Blocks(title="Lensieve") as demo:
        gr.Markdown("# Lensieve")

        with gr.Column():
            gr.Markdown("### Request")
            query = gr.Textbox(
                placeholder="Search your photos...",
                lines=1,
                max_lines=3,
                show_label=False,
                container=False,
            )

            gr.Markdown("### Response")
            text_output = gr.Markdown()

            gallery_output = gr.Gallery(
                columns=6,
                rows=2,
                height=450,
                object_fit="contain",
                show_label=False,
                container=False,
            )

        query.submit(
            fn=run_query,
            inputs=query,
            outputs=[text_output, gallery_output],
        )

    return demo
