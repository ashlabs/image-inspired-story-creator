"""Launch the image-inspired story creator on localhost."""

import logging
import threading
from pathlib import Path

import gradio as gr

from app_state import Session

LOGGER = logging.getLogger(__name__)
MODEL_LOCK = threading.Lock()
_models = None


def get_models():
    """Return shared local models; caller must hold MODEL_LOCK."""
    global _models
    if _models is None:
        from app_runtime import LocalModels
        _models = LocalModels(Path(__file__).resolve().parent)
    return _models


def generate_caption(session):
    """Run caption inference and discard results from superseded requests."""
    try:
        revision, image = session.begin_caption()
    except ValueError as error:
        gr.Warning(str(error))
        return
    try:
        with MODEL_LOCK:
            caption = get_models().caption(image)
        session.finish_caption(revision, caption)
    except Exception as error:
        LOGGER.exception("Caption generation failed")
        session.fail(revision, f"Caption could not be generated: {error}")


def approve_premise(text, session):
    """Approve exactly the currently displayed premise."""
    try:
        session.approve(text)
    except ValueError as error:
        gr.Warning(str(error))
        return


def generate_story(text, session):
    """Send only an approved premise string to the separate storyteller."""
    try:
        revision, approved = session.begin_story(text)
    except ValueError as error:
        gr.Warning(str(error))
        return
    try:
        with MODEL_LOCK:
            story, details = get_models().story(approved)
        session.finish_story(revision, story, details)
    except Exception as error:
        LOGGER.exception("Story generation failed")
        session.fail(revision, f"Story could not be generated: {error}")


def render(session):
    """Display only current session state, including during model execution."""
    pending, approved, story, status, details = session.snapshot()
    return (
        gr.skip() if pending is None else pending,
        approved, story, status, details,
    )


def build_app():
    """Build the local photo, approval, and fictional-story interface."""
    with gr.Blocks(title="Image-Inspired Story Creator") as demo:
        gr.Markdown(
            "# Image-Inspired Story Creator\n"
            "Upload a photograph, correct its caption, approve your premise, "
            "and generate a fictional story."
        )
        session = gr.State(value=Session)
        with gr.Row():
            with gr.Column():
                photo = gr.Image(type="pil", sources=["upload"], label="Photograph")
                caption_button = gr.Button("Generate caption")
                gr.Markdown(
                    "Captions can misidentify objects and details. "
                    "Check the photograph and correct the text before approval."
                )
            with gr.Column():
                premise = gr.Textbox(label="Editable premise", lines=4, max_lines=10)
                approve_button = gr.Button("Approve premise")
                approved = gr.Textbox(label="Approved premise", interactive=False, lines=3)
                story_button = gr.Button("Generate story", variant="primary")
        status = gr.Textbox(label="Status", interactive=False, value="Upload a photograph to begin.")
        gr.Markdown("## Fictional story\nInvented events and characters are fiction.")
        story = gr.Textbox(label="Fictional story", interactive=False, lines=9)
        details = gr.Textbox(label="Generation details", interactive=False)

        # Edits bypass the model queue so approval and old results clear promptly.
        photo.change(
            lambda image, state: state.replace_image(image),
            inputs=[photo, session], outputs=None, queue=False,
        )
        premise.input(
            lambda text, state: state.edit_premise(text),
            inputs=[premise, session], outputs=None, queue=False,
        )
        approve_button.click(
            approve_premise, inputs=[premise, session], outputs=None, queue=False,
        )
        caption_button.click(
            generate_caption, inputs=[session], outputs=None,
            concurrency_id="local_models", concurrency_limit=1,
        )
        story_button.click(
            generate_story, inputs=[premise, session], outputs=None,
            concurrency_id="local_models", concurrency_limit=1,
        )
        # A single renderer avoids old model callbacks overwriting newer UI state.
        timer = gr.Timer(0.5)
        timer.tick(
            render, inputs=[session], outputs=[premise, approved, story, status, details],
            queue=False, show_progress="hidden",
        )
    return demo


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_app().queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1", share=False, inbrowser=True,
    )
