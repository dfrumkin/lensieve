import os

# Assume no connection to the internet; do not want to share data.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

from typing import Any

import gradio as gr

from lensieve.agent.photo_agent import PhotoAgent
from lensieve.image import load_image
from lensieve.retrieval.schema import SearchResult


def build_app(agent: PhotoAgent) -> gr.Blocks:
    def run_query(user_query: str) -> tuple[str, Any]:
        user_query = user_query.strip()

        if not user_query:
            return "Enter a query.", gr.update(value=[])

        try:
            result = agent.run_once(user_query)

            if isinstance(result, str):
                text = result
                images = []

            elif isinstance(result, SearchResult):
                hits = result.hits
                num_hits = len(hits)
                text = "No matches found." if num_hits == 0 else f"Showing top {num_hits} matches."
                # Return PIL images to support multiple formats
                # Could convert to JPEG and cache, but then would need to maintain the cache
                images = [(load_image(hit.path), hit.path.name) for hit in hits]
            else:
                text = f"Internal error: result type is {type(result).__name__}"
                images = []

            return text, gr.update(value=images)

        except Exception as e:
            return f"Error: {e}", gr.update(value=[])

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
