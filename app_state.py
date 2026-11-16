"""Per-session approval and result state, independent of model libraries."""

from threading import RLock


class Session:
    """Keep one browser session's image, premise, approval, and story isolated."""

    def __init__(self):
        self.lock = RLock()
        self.revision = 0
        self.image = None
        self.premise = ""
        self.approved = ""
        self.story = ""
        self.details = ""
        self.status = "Upload a photograph to begin."
        self.pending_premise = None

    def __deepcopy__(self, memo):
        """Give Gradio a fresh session instead of copying a thread lock."""
        return Session()

    def _invalidate(self):
        self.revision += 1
        self.approved = ""
        self.story = ""
        self.details = ""

    def replace_image(self, image):
        """Replace an upload and invalidate every result from the prior image."""
        with self.lock:
            self._invalidate()
            self.image = image.copy() if image is not None else None
            self.premise = ""
            self.pending_premise = ""
            self.status = (
                "Image ready. Select Generate caption."
                if image is not None else "Upload a photograph to begin."
            )

    def begin_caption(self):
        """Return a revision and image snapshot for a caption request."""
        with self.lock:
            if self.image is None:
                raise ValueError("Upload a photograph first.")
            self._invalidate()
            self.status = "Generating caption…"
            return self.revision, self.image.copy()

    def finish_caption(self, revision, caption):
        """Publish a caption only if its image and premise are still current."""
        with self.lock:
            if revision != self.revision:
                return False
            self.premise = caption
            self.pending_premise = caption
            self.status = "Review and correct the caption, then approve the premise."
            return True

    def edit_premise(self, text):
        """Invalidate approval and story whenever the premise changes."""
        with self.lock:
            if text != self.premise:
                self._invalidate()
                self.premise = text
                self.pending_premise = None
                self.status = "Premise changed. Review it and approve again."

    def approve(self, text):
        """Atomically save and approve the textbox value submitted on the click."""
        with self.lock:
            if self.image is None:
                raise ValueError("Upload a photograph first.")
            if not text.strip():
                raise ValueError("Enter a premise before approving it.")
            if len(text) > 2000:
                raise ValueError("Keep the premise within 2,000 characters.")
            # The click carries the authoritative textbox snapshot. An input
            # callback may still be in flight, so save and approve it together.
            self._invalidate()
            self.premise = text
            self.pending_premise = None
            self.approved = text
            self.status = "Premise approved. Select Generate story."

    def begin_story(self, visible_premise):
        """Return approved text only; no image or hidden caption is returned."""
        with self.lock:
            if not self.approved or visible_premise != self.approved:
                raise ValueError("Approve the current premise before generating a story.")
            self.revision += 1
            self.story = ""
            self.details = ""
            self.status = "Generating fictional story…"
            return self.revision, self.approved

    def finish_story(self, revision, story, details):
        """Discard late results if upload, premise, or approval changed."""
        with self.lock:
            if revision != self.revision:
                return False
            self.story = story
            self.details = details
            self.status = "Story ready. Editing the premise clears this story."
            return True

    def fail(self, revision, message):
        """Show an error only when it belongs to the active request."""
        with self.lock:
            if revision == self.revision:
                self.status = message

    def snapshot(self):
        """Return display values and consume any pending premise update."""
        with self.lock:
            pending = self.pending_premise
            self.pending_premise = None
            return pending, self.approved, self.story, self.status, self.details
