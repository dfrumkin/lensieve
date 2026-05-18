import os

# Assume no connection to the internet; do not want to share data.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

from typing import Any

import gradio as gr
from PIL.Image import Image

from lensieve.agent.photo_agent import PhotoAgent
from lensieve.app.grouping import ImageGroup, group_hits_by_similarity
from lensieve.image import load_image
from lensieve.retrieval.schema import SearchResult


def _get_image_data(group: ImageGroup) -> tuple[Image, str]:
    path = group.representative.path
    image = load_image(path)
    name = group.representative.path.name
    num_extra = len(group.items) - 1
    if num_extra > 0:
        name += f" +{num_extra}"
    return image, name


def _group(result: SearchResult, threshold: float) -> tuple[list[ImageGroup], list[tuple[Image, str]], str]:
    groups = group_hits_by_similarity(result, threshold)
    images = [_get_image_data(group) for group in groups]
    num_hits = len(images)
    text = "No matches found." if num_hits == 0 else f"Showing top {num_hits} match{'' if num_hits == 1 else 'es'}."
    return groups, images, text


def build_app(agent: PhotoAgent) -> gr.Blocks:
    def run_query(user_query: str, threshold: float) -> tuple[SearchResult | None, list[ImageGroup], str, Any, Any]:
        user_query = user_query.strip()

        if not user_query:
            return None, [], "Enter a query.", [], []

        try:
            result = agent.run_once(user_query)

            if isinstance(result, str):
                text = result
                result = None
                images = []
                groups = []

            elif isinstance(result, SearchResult):
                groups, images, text = _group(result, threshold)
            else:
                text = f"Internal error: result type is {type(result).__name__}"
                result = None
                images = []
                groups = []

            return result, groups, text, images, []

        except Exception as e:
            return None, [], f"Error: {e}", [], []

    def show_group(
        groups: list[ImageGroup],
        evt: gr.SelectData,
    ) -> list[tuple[Image, str]]:
        idx = evt.index

        if idx is None or idx >= len(groups):
            return []

        return [(load_image(item.path), item.path.name) for item in groups[idx].items] if groups else []

    def regroup_existing_results(
        result: SearchResult | None,
        threshold: float,
    ) -> tuple[list[ImageGroup], str, list[tuple[Image, str]], list]:
        if result is None:
            return [], "", [], []

        groups, images, text = _group(result, threshold)
        return groups, text, images, []

    with gr.Blocks(title="Lensieve") as demo:
        gr.Markdown("# Lensieve")

        gr.Markdown("### Request")

        query = gr.Textbox(
            placeholder="Search your photos...",
            lines=1,
            max_lines=3,
            show_label=False,
            container=False,
        )

        group_slider = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=1.0,
            step=0.01,
            label="Grouping similarity threshold",
            info="Higher = stricter grouping; lower = more images grouped together.",
        )

        gr.Markdown("### Response")
        text_output = gr.Markdown()

        results_gallery = gr.Gallery(
            columns=6,
            rows=2,
            height=600,
            object_fit="contain",
            show_label=False,
            container=False,
        )

        group_gallery = gr.Gallery(
            label="Selected group",
            columns=6,
            height=600,
            show_label=True,
        )

        result_state = gr.State(None)
        groups_state = gr.State([])

        query.submit(
            fn=run_query,
            inputs=[query, group_slider],
            outputs=[result_state, groups_state, text_output, results_gallery, group_gallery],
        )

        group_slider.release(
            fn=regroup_existing_results,
            inputs=[result_state, group_slider],
            outputs=[groups_state, text_output, results_gallery, group_gallery],
        )

        results_gallery.select(
            show_group,
            inputs=[groups_state],
            outputs=[group_gallery],
        )

    return demo
