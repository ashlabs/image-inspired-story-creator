"""Exercise approval boundaries and late-result handling without model weights."""

import copy
import unittest

from app_state import Session


class ImageStub:
    """Minimal upload snapshot for session tests."""

    def copy(self):
        return ImageStub()


class SessionTests(unittest.TestCase):
    def ready(self):
        state = Session()
        state.replace_image(ImageStub())
        state.edit_premise("A cat on a horse.")
        return state

    def test_requires_image(self):
        with self.assertRaises(ValueError):
            Session().begin_caption()

    def test_requires_approval(self):
        with self.assertRaises(ValueError):
            self.ready().begin_story("A cat on a horse.")

    def test_story_receives_exact_approved_text(self):
        state = self.ready()
        state.approve(state.premise)
        revision, text = state.begin_story(state.premise)
        self.assertIsInstance(text, str)
        self.assertEqual(text, "A cat on a horse.")
        self.assertTrue(state.finish_story(revision, "Fiction", "2 words"))

    def test_edit_invalidates_completed_story_and_approval(self):
        state = self.ready()
        state.approve(state.premise)
        revision, _ = state.begin_story(state.premise)
        state.finish_story(revision, "Story", "details")
        state.edit_premise("A cat on a table.")
        self.assertEqual((state.story, state.approved), ("", ""))

    def test_edit_discards_inflight_story(self):
        state = self.ready()
        state.approve(state.premise)
        revision, _ = state.begin_story(state.premise)
        state.edit_premise("A cat on a table.")
        self.assertFalse(state.finish_story(revision, "Old story", ""))
        self.assertEqual(state.story, "")

    def test_upload_discards_old_caption(self):
        state = self.ready()
        revision, _ = state.begin_caption()
        state.replace_image(ImageStub())
        self.assertFalse(state.finish_caption(revision, "Old caption"))
        self.assertEqual(state.premise, "")

    def test_premise_edit_discards_old_caption(self):
        state = self.ready()
        revision, _ = state.begin_caption()
        state.edit_premise("My own premise.")
        self.assertFalse(state.finish_caption(revision, "Old caption"))
        self.assertEqual(state.premise, "My own premise.")

    def test_approve_saves_click_text_before_input_callback(self):
        state = self.ready()
        revision, _ = state.begin_caption()
        state.finish_caption(revision, "A goalie is kicking a ball.")
        corrected = "A player kicks a ball toward a goalkeeper."
        state.approve(corrected)
        self.assertEqual(state.premise, corrected)
        self.assertEqual(state.approved, corrected)
        self.assertIsNone(state.snapshot()[0])
        # The delayed input callback contains the same submitted text.
        state.edit_premise(corrected)
        self.assertEqual(state.approved, corrected)
        self.assertEqual(state.begin_story(corrected)[1], corrected)

    def test_approval_discards_inflight_caption(self):
        state = self.ready()
        revision, _ = state.begin_caption()
        state.approve("A user-written premise.")
        self.assertFalse(state.finish_caption(revision, "Late caption"))
        self.assertEqual(state.approved, "A user-written premise.")

    def test_new_approval_discards_inflight_story(self):
        state = self.ready()
        state.approve(state.premise)
        revision, _ = state.begin_story(state.premise)
        state.approve("A revised premise from the textbox.")
        self.assertFalse(state.finish_story(revision, "Old story", ""))
        self.assertEqual(state.story, "")

    def test_mismatched_visible_premise_cannot_generate(self):
        state = self.ready()
        state.approve(state.premise)
        with self.assertRaises(ValueError):
            state.begin_story("Different text")

    def test_empty_or_long_premise_cannot_be_approved(self):
        state = self.ready()
        for text in (" ", "x" * 2001):
            state.edit_premise(text)
            with self.assertRaises(ValueError):
                state.approve(text)

    def test_sessions_are_isolated(self):
        first = self.ready()
        second = copy.deepcopy(first)
        self.assertEqual(second.premise, "")
        first.approve(first.premise)
        self.assertEqual(second.approved, "")

    def test_clear_upload_invalidates_inflight_story(self):
        state = self.ready()
        state.approve(state.premise)
        revision, _ = state.begin_story(state.premise)
        state.replace_image(None)
        self.assertFalse(state.finish_story(revision, "Old story", ""))
        self.assertIsNone(state.image)
        self.assertEqual((state.story, state.approved, state.premise), ("", "", ""))

    def test_render_consumes_premise_update_once(self):
        state = self.ready()
        revision, _ = state.begin_caption()
        state.finish_caption(revision, "New caption")
        self.assertEqual(state.snapshot()[0], "New caption")
        self.assertIsNone(state.snapshot()[0])


if __name__ == "__main__":
    unittest.main()
