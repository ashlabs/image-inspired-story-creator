# Release evaluation procedure

The selected model and integrated app are evaluated without further caption-model
experiments. The application is a prototype requiring correction and approval of
caption text. Target failures remain part of the result.

## Story suite and runtime

Run `final_evaluation.ipynb` from the repository root in the existing project
virtual environment. Stop the app and other model kernels first. Sections 1–3
verify the selected artifacts, freeze the implementation hashes, generate the
fixed 50-premise story suite, and describe the narrative rating guide.

The suite contains 20 minimally edited human reference captions from development
images, 20 fictional elaborations of those premises, and 10 variants containing
quoted instruction-like sign text. It is a development suite, not an unseen
final-test claim. Added details are fictional edits, not observations about the
photographs. No final-test photograph or reference determines the suite.

Saved story outputs include actual word count, latency, and unfilled review
fields. Review premise preservation and coherence separately from length. Keep
`story_outputs_v1.json` unchanged and put confirmed review judgments in a separate
file. Targets remain 90% premise preservation, 80% coherence at least 4/5, and
95% length compliance within the application's bounded retry.

The notebook excludes the first story call from warm latency. Caption latency
uses a development-image warm-up; measured held-out calls include preprocessing
and model inference but exclude disk decoding and file hashing. Warm story p95
is computed over 50 requests and caption p95 over 857 images. Report cold
initialization separately. Process RSS and an end-of-run MPS driver counter are
not a complete measurement of combined unified-memory working set or swapping.

## Held-out captions

After reviewing the frozen setup, set `RUN_FINAL_CAPTIONS = True` in section 4.
Generate captions for the 857 frozen final-test IDs once. Report unsmoothed
corpus BLEU-1 and BLEU-4 against all human references using the existing tokenizer.
The existing confirmed 200-image development review remains the grounding
assessment; no additional 857-image manual labeling exercise is introduced.

The original 80% caption criterion concerns main subject/action correctness.
ResNet50's confirmed development result is 84.5%. The major unsupported-object or
action criterion is at most 10%; the measured 38.5% fails. Overall no-error pass
is a separate measure at 37.0%. Comparing ResNet18 and ResNet50 does not establish
superiority over a newly computed ResNet50 nearest-neighbor retrieval baseline.
That comparison remains unmeasured and must not be presented as passed.

## Saved outputs and resuming

Reports are written under `reports/final_evaluation_v1/`. Per-item progress is
kept under the ignored `data/processed/final_evaluation_v1/` directory. Reruns reuse
saved completed outputs rather than regenerating them. An incomplete or corrupt
progress record raises an error for inspection. Changed frozen source, model,
configuration, or suite hashes also raise an error rather than overwriting a run.

Do not modify the model, preprocessing, prompts, decoding, or thresholds based
on held-out results. If results motivate tuning, reserve a new final-test set.
Do not infer narrative correctness or image grounding from BLEU or length alone.
