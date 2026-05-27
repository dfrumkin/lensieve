import os

# Assume no connection to the internet; do not want to share data.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

from collections.abc import Iterator
from typing import Any

import gradio as gr
from PIL.Image import Image

from lensieve.agent.photo_agent import PhotoAgent
from lensieve.image import load_image
from lensieve.retrieval.grouping import ImageGroup, ImageGroupItem, group_hits_by_similarity
from lensieve.retrieval.schema import SearchResult

GalleryData = list[tuple[Image, str]]


def _get_image_group_data(group: ImageGroup) -> tuple[Image, str]:
    path = group.representative.path
    image = load_image(path)

    label = path.name
    num_extra = len(group.items) - 1
    if num_extra > 0:
        label += f" +{num_extra}"

    return image, label


def _get_image_group_item_data(item: ImageGroupItem) -> tuple[Image, str]:
    path = item.hit.path
    image = load_image(path)

    score = item.similarity_to_representative
    label = f"{path.name} ({score:.2f})"

    return image, label


def _build_grouped_gallery(
    result: SearchResult,
    threshold: float,
) -> tuple[list[ImageGroup], GalleryData, str]:
    groups = group_hits_by_similarity(result, threshold)
    images = [_get_image_group_data(group) for group in groups]

    num_groups = len(groups)
    num_hits = len(result.hits)

    if num_hits == 0:
        text = "No matches found."
    elif num_groups == num_hits:
        text = f"Showing {num_hits} match{'' if num_hits == 1 else 'es'}."
    else:
        text = (
            f"Showing {num_groups} group{'' if num_groups == 1 else 's'} "
            f"from {num_hits} match{'' if num_hits == 1 else 'es'}."
        )

    return groups, images, text


def _build_empty_output(
    msg: str,
) -> tuple[
    SearchResult | None,
    list[ImageGroup],
    str,
    Any,
    Any,
    Any,
]:
    return (
        None,
        [],
        msg,
        gr.update(value=[], visible=False),
        gr.update(value=[], visible=False),
        gr.update(visible=False),
    )


RunQueryOutput = tuple[
    SearchResult | None,
    list[ImageGroup],
    str,
    Any,
    Any,
    Any,
]


def build_app(agent: PhotoAgent) -> gr.Blocks:
    def run_query(
        user_query: str,
        threshold: float,
    ) -> Iterator[RunQueryOutput]:
        user_query = user_query.strip()

        if not user_query:
            yield _build_empty_output("Enter a query.")
            return

        yield _build_empty_output("<span style='color: gray'>Thinking...</span>")

        try:
            result = agent.run_once(user_query)
        except Exception as e:
            yield _build_empty_output(f"Error: {e}")
            return

        if isinstance(result, SearchResult):
            groups, images, text = _build_grouped_gallery(result, threshold)
            has_hits = len(images) > 0
            yield (
                result,
                groups,
                text,
                gr.update(value=images, visible=has_hits),
                gr.update(value=[], visible=False),
                gr.update(value=1.0, visible=has_hits),
            )
            return

        if isinstance(result, str):
            yield _build_empty_output(result)
            return

        yield _build_empty_output(f"Internal error: result type is {type(result).__name__}")
        return

    def show_group(
        groups: list[ImageGroup],
        evt: gr.SelectData,
    ) -> Any:
        idx = evt.index

        if idx is None or idx >= len(groups):
            return gr.update(value=[], visible=False)

        group = groups[idx]
        return gr.update(
            value=[_get_image_group_item_data(item) for item in group.items],
            visible=True,
        )

    def regroup_existing_results(
        result: SearchResult | None,
        threshold: float,
    ) -> tuple[list[ImageGroup], str, Any, Any]:
        if result is None:
            return [], "", gr.update(value=[], visible=False), gr.update(value=[], visible=False)

        groups, images, text = _build_grouped_gallery(result, threshold)
        return groups, text, gr.update(value=images, visible=len(images) > 0), gr.update(value=[], visible=False)

    with gr.Blocks(title="Lensieve") as demo:
        gr.Markdown("# Lensieve")

        gr.Markdown("### Request")

        query = gr.Textbox(
            placeholder="Search your photos or ask about your photo metadata...",
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
            info="Used only for image search results. Higher = stricter grouping; lower = more images grouped together.",
            visible=False,
        )

        gr.Markdown("### Response")

        text_output = gr.Markdown()

        results_gallery = gr.Gallery(
            label="Search results",
            columns=6,
            rows=2,
            height=600,
            object_fit="contain",
            show_label=True,
            container=False,
            visible=False,
        )

        group_gallery = gr.Gallery(
            label="Selected group",
            columns=6,
            height=600,
            object_fit="contain",
            show_label=True,
            visible=False,
        )

        image_result_state = gr.State(None)
        groups_state = gr.State([])

        query.submit(
            fn=run_query,
            inputs=[query, group_slider],
            outputs=[
                image_result_state,
                groups_state,
                text_output,
                results_gallery,
                group_gallery,
                group_slider,
            ],
        )

        group_slider.release(
            fn=regroup_existing_results,
            inputs=[image_result_state, group_slider],
            outputs=[
                groups_state,
                text_output,
                results_gallery,
                group_gallery,
            ],
        )

        results_gallery.select(
            fn=show_group,
            inputs=[groups_state],
            outputs=[group_gallery],
        )

    return demo
