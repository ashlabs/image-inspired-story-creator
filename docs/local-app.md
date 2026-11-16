# Local application

The application combines the trained ResNet50 caption decoder with the pinned,
locally cached Qwen2.5-1.5B-Instruct storyteller. Caption output is a draft that
requires review and correction. The model's confirmed development grounding pass
rate is 37.0%; it does not meet the original automatic-caption quality targets.

## Launch

From the repository root, with the project's native Apple Silicon virtual
environment active:

```bash
python -m pip install -r requirements-app.txt
python app.py
```

The browser opens a localhost address, normally http://127.0.0.1:7860. The app
uses `127.0.0.1` and `share=False`. Stop it with Ctrl+C in the terminal.
The existing environment already contains these dependencies; reinstalling is
unnecessary if it has not changed. Close training notebook kernels before launch
to release their model allocations.

Required local assets from the completed experiments:

- `data/processed/caption_training/resnet50_v1/best.pt`
- ResNet50 V2 weights in the torchvision checkpoint cache
- Qwen model and tokenizer revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
  in the Hugging Face cache

No training data or visual-feature caches are needed for inference. Local
checkpoint and manifest checksums are verified before caption inference. Missing
weights produce an error rather than starting a new download. The first caption
loads the caption components; the first story also loads the story model, so
initial requests take longer than subsequent requests.

## Use

1. Upload a photograph and select **Generate caption**.
2. Correct the editable premise against the photograph.
3. Select **Approve premise** and check the approved text.
4. Select **Generate story**.

The storyteller receives only approved text. Uploads and premise edits invalidate
approval, clear previous stories, and prevent late model results from replacing
current session state. The interface refreshes its status approximately every
half second. Sessions do not share images or premises; model weights are shared
and inference requests are serialized.

Stories are labeled fiction. Word count uses whitespace-separated words. A story
outside 100–150 words receives at most one retry; if the retry still misses the
target, its actual length is shown. The initial prompt asks for a complete incident
with one consistent outcome. The retry receives the actual draft and measured word
count, then expands or shortens it toward 120–135 words while checking consistency.
Stories are not mechanically truncated.
Generated fiction is not a source of factual information about photographed
people. Prompt instructions encourage premise preservation; they do not guarantee
that every generated statement follows the premise.

## Validation

```bash
python -m unittest discover -s tests -p 'test_app_state.py' -v
```

These tests cover approval, premise edits, upload changes, session isolation,
visible-text matching, and discarding stale caption/story results. Actual caption
and story inference must be checked on the Mac with the recorded local weights.

For the first local run, use a development photograph, generate and correct a
caption, approve it, then generate a story. Edit the premise and check that the
approved text and story clear. Upload a second image and check the premise also
resets. Keep the final-test photographs unused until the implementation is frozen.

## Local walkthrough results

A local Mac walkthrough was completed on November 16, 2026, covering the
application changes through commit `9797939`.

| Check | Observed result |
| --- | --- |
| Upload and caption generation | A caption was generated for an uploaded photograph. |
| Premise editing and approval | The corrected premise could be approved after the synchronization fix. |
| Fictional-story generation | A complete story was displayed with generation details. |
| Editing the premise | Approved text and the previous story cleared, as confirmed by the author. |
| Uploading another photograph | Editable premise, approved premise, and story cleared, as confirmed by the author. |

One reported story contained 105 words, completed in one attempt, and took
1.8 seconds according to the app's generation details. Earlier examples missed
the length target, including a 97-word result after two attempts. The revised
retry supplies the actual draft and word count; its overall success rate has
not yet been measured.

Story coherence remains a limitation. In the 105-word example, a player trying
to score reacts with relief after missing, without an explanation. Successful
interface operation does not establish consistent narrative quality.

The automated session suite contains 15 passing tests, including approval from
a textbox snapshot before its edit callback finishes. Approval and rendering
were also checked through Gradio 6.31.0's callback processing with simulated
delayed synchronization. These automated checks do not replace the local
walkthrough or the planned final quality, latency, and memory evaluation.
