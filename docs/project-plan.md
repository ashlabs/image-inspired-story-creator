# Image-Inspired Story Creator — Project Plan

## 1. Goal and scope

Image-conditioned caption generation followed by user-controlled fictional
storytelling. Caption review and premise approval precede story generation.

Scope: one photograph, one editable premise, and one fictional story. Story-model
fine-tuning, recurring characters, and illustrated narratives are excluded.

### Initial release

| Property | Specification |
|---|---|
| Execution | Local inference on Apple Silicon using MPS |
| Interface | Browser-based Gradio application |
| Input | One photograph and an editable text premise |
| Output | One fictional story of 100–150 words |
| Hosting | Localhost |
| Excluded features | Public hosting, accounts, galleries, voice, and multi-image narratives |

## 2. User flow and application behavior

1. Upload a photograph.
2. Generate and display an automated caption.
3. Let the user edit the caption into a story premise.
4. Generate a story only when the user selects **Generate Story**.
5. Display the approved premise and clearly labeled fictional story separately
   from the automated caption.

The storyteller receives only the current approved premise. It receives no
photograph, image features, hidden original caption, or earlier story context.
User edits can deliberately depart from the image; those additions are not
presented as verified visual details.

A premise edit marks the existing story outdated. Regeneration is required to use
the new text. If the premise changes while generation is running, the completed
response is also marked outdated. Uploading another image clears the previous
caption, premise, and story.

Captions should describe visible content without inferring identity or sensitive
attributes. Names, motives, dialogue, and backstory belong in the fictional story.

## 3. Architecture

| Component | Implementation |
|---|---|
| Image encoder | Frozen ResNet18 with pretrained ImageNet weights |
| Visual projection | Trainable mapping from 512 features to decoder width 256 |
| Caption decoder | Randomly initialized Transformer implemented and trained in PyTorch |
| Story generator | Separate pretrained local language model |
| Application | Gradio interface with upload validation, editable state, and inference |

The caption decoder is trained from random initialization. The image encoder
and storyteller use pretrained weights. The encoder supplies visual features,
and the storyteller generates fiction from the approved premise. Caption quality
and story quality are measured separately.

### Caption model

Initial decoder configuration:

- A 7 × 7 spatial feature grid: 49 positions with 512 features each.
- A trainable projection to width 256.
- Learned token and position embeddings.
- Two Transformer decoder blocks with four attention heads.
- Feed-forward width 1,024 and dropout 0.1.
- A training-only vocabulary capped initially at 8,000 tokens.
- Teacher-forced next-token prediction with cross-entropy loss, ignoring padding.
- AdamW with an initial learning rate of 0.0003.
- Checkpoint selection using validation results.
- Autoregressive caption generation.

Vocabulary, caption-length policy, and epoch budget: determined from training-data
inspection and the real-data pilot before the baseline training run.

Preprocessing evaluation: default ResNet18 center crop versus a scene-preserving
transform. Caption-relevant content outside the crop is a specific failure case.
The selected transform applies consistently to training, evaluation, and inference.

### Story model

Story-model candidate: Qwen2.5-1.5B-Instruct, local MPS inference in float16.
Experiment records include the model revision, prompt, and decoding settings.

Initial findings: omitted premise details and weak narrative logic. Selection
for application integration depends on results from the fixed story review suite.
Story-model training and fine-tuning are excluded.

## 4. Data

Dataset: COCO 2017 photographs and human-written captions.
Initial image eligibility: recorded license ID 4, CC BY 2.0.
Caption annotation terms: COCO CC BY 4.0.

| Role | Planned images | Source |
|---|---:|---|
| Training | 16,000 | train2017 candidates |
| Checkpoint validation | 2,000 | Separate train2017 candidates |
| Fixed development review | 200 | Separate train2017 candidates |
| Reserve | 1,270 | Remaining train2017 candidates |
| Final test | Up to 857 | val2017 candidates |

These allocations depend on successful downloads, duplicate removal, and the
attribution audit. Save the final IDs, selection rules, random seed, and checksums
in a versioned split manifest.

Keep every image's captions in the same split. Group exact and near-duplicate
images before splitting. Fit vocabulary and token statistics on training captions
only. Do not inspect final-test content or use it to select models or prompts.
Image 522418 has already been inspected and remains outside final testing.

Retain source links and collect attribution details for published examples.
Complete attribution for the photograph displayed in the feasibility notebook.
Check whether license filtering changes the distribution of scenes and activities.

## 5. Evaluation

### Caption quality

Use the fixed 200-image development set and define the rating guide before scoring.

A caption passes when it captures the main subject or action without a material
invented claim. Track unsupported objects, actions, attributes, and relationships
as separate failure categories.

Targets:

- At least 80% of captions pass the main-subject/action check.
- No more than 10% contain a major unsupported object or action.
- A 10-percentage-point improvement in human pass rate over nearest-neighbor
  retrieval of training captions using the same frozen image features.

Use paired comparisons and report uncertainty. If the interval includes no
improvement, the comparison is inconclusive. Record evaluator identity, ratings,
and raw outputs so that the judgment can be reviewed.

Use validation loss and a documented caption-overlap metric as secondary
diagnostics. Neither fluent text nor a high overlap score establishes image
grounding. Test shuffled or zeroed image features to check whether the decoder
uses visual information.

### Story quality

Use a fixed development suite of 50 premises, including lightly edited premises,
heavily edited premises, and instruction-like text that should be treated as data.
Define an anchored coherence rating guide before scoring.

Targets:

- At least 90% preserve explicit premise facts without contradiction.
- At least 80% score 4/5 or better for coherence.
- At least 95% produce 100–150 words within one bounded retry.
- All outputs are labeled fictional.
- All edit-control cases use only the current approved premise.

Count words using a documented whitespace-based rule initially. If generation
cannot meet the length requirement within the retry limit, return a useful error
instead of silently truncating the text.

Evaluate stories from corrected premises to isolate story-model behavior. Also
evaluate stories from generated captions to measure how caption errors propagate.
An appealing story does not compensate for an inaccurate caption.

### Runtime and reliability

Measure on the target Mac:

- A projected baseline caption-training run of no more than eight hours.
- A combined working-set target below 18 GiB, without sustained swapping.
- Warm caption p95 latency of at most five seconds.
- Warm story p95 latency of at most 30 seconds, over at least 30 requests.
- Cold startup time reported separately.

The synthetic benchmark does not establish real training speed or peak memory.
Measure both with actual data before committing to a full run.

Validate uploads and handle unsupported formats, corrupt files, excessive file
size or resolution, empty premises, missing artifacts, device errors, and generation
failures. Check artifact reload and setup in a clean environment.

## 6. Experiment budget

Run one compact caption baseline and at most two hypothesis-driven changes.
For each change, record the observed failure, hypothesis, setting changed, and
comparison needed to judge the result.

If the caption quality target remains unmet after that budget, stop and review
scope. Replacing the trained caption model with a pretrained captioner would be
a change to the project architecture, not completion of this plan.

If the compute budget is exceeded, allow one batch-size or cache adjustment,
then review scope.

Freeze the checkpoint, preprocessing, prompts, decoding settings, and thresholds
before final testing. Evaluate the final set once and report failures. If those
results drive further tuning, reserve a new final set.

## 7. Milestones

| Milestone | Completion criteria |
|---|---|
| Feasibility | Dataset access, local component execution, and limitations documented; notebook runs from a fresh kernel |
| Project baseline | Scope, architecture, evaluation targets, and milestones reviewed |
| Data and baselines | Audited downloads, fixed splits, vocabulary, and retrieval baseline |
| Caption correctness | Causal and padding-mask checks, tiny-subset overfit, and checkpoint reload |
| Caption quality | Bounded experiments, scorecard, and failure analysis |
| Story integration | Premise adherence, length, coherence, and edit-control evaluation |
| Local application | Complete user flow, state handling, errors, timing, and artifact loading |
| Release | Frozen final evaluation, reproducible setup, documented artifacts, and working local launch |

## 8. Development workflow

Version control: stable `main`, focused development branches, milestone commits,
and pull-request review before merging.

Model development environment: Jupyter.
Stage documentation: purpose, inputs, outputs, design decisions, and observed results.
Progression criterion: review of the current stage's results.
Function documentation: parameters, return values, tensor shapes, and errors.

Development records: experiment and decision logs.
Application integration: stable inference code extracted into Python modules.

Git exclusions: datasets, environments, caches, credentials, and large weights.
Model artifacts: separately versioned with hashes, source revisions, configuration,
vocabulary, preprocessing, and applicable terms.
