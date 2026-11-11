# Image-Inspired Story Creator

Image captioning and user-controlled fictional storytelling with local inference.

## Initial release

| Property | Specification |
|---|---|
| Execution | Local inference on Apple Silicon using MPS |
| Interface | Browser-based Gradio application |
| Input | One photograph and an editable text premise |
| Output | One fictional story of 100–150 words |
| Hosting | Localhost |

## Application flow

Image upload → automated caption → premise editing and approval → explicit
**Generate Story** action → fictional story.

## How it works

1. A frozen ResNet18 encoder extracts visual features from the photograph.
2. A Transformer decoder trained on image-caption pairs generates a caption.
3. The user reviews and edits the caption into an approved premise.
4. A separate local language model generates a story from that premise.

The caption and story are displayed separately. The storyteller receives only the
approved premise. It does not receive the photograph, visual features, or original
caption. Details added through editing are part of the story premise, not verified
facts about the image.

Editing the premise marks the existing story outdated. Generate again to use the
changes. Every story is labeled as fiction.

## Architecture

| Component | Approach |
|---|---|
| Image encoder | Frozen ResNet18 with pretrained ImageNet weights |
| Caption decoder | Randomly initialized Transformer, implemented and trained in PyTorch |
| Story generator | Local Qwen2.5-1.5B-Instruct inference |
| Interface | Planned Gradio application running locally on the Mac |

The caption decoder is trained from random initialization. The image encoder
and storyteller use pretrained weights. The complete system is not trained
end to end.

## Development status

Dataset and runtime feasibility checks are complete. The notebook runs on an
Apple M5 Pro using MPS, including pretrained image-feature extraction, synthetic
Transformer training steps, and local story generation.

Pending milestones: dataset splits, caption training, model evaluation, and
application implementation. Initial Qwen samples omit premise details and contain
narrative weaknesses. Story-model selection depends on development evaluation.

## Dataset

The dataset consists of COCO 2017 photographs and human-written captions. The initial selection
includes image records marked CC BY 2.0: 19,470 training-source candidates and
857 final-test candidates. COCO caption annotations have separate CC BY 4.0 terms.

Published examples require source links and attribution details. Attribution for
the photograph displayed in the feasibility notebook is incomplete.
Dataset archives, environments, caches, and model weights are excluded from Git.

## Feasibility notebook

- `feasibility/dataset_feasibility.ipynb`: dataset inspection and runtime checks.
- `feasibility/requirements-feasibility.txt`: package versions used for these checks.

Run the notebook from `feasibility/` with `annotations_trainval2017.zip` in the same
folder. The pinned Qwen model must already be cached; download instructions are in
the notebook. ResNet18 weights and the sample photograph may require network access.

The decoder benchmark uses synthetic batches. Its timing excludes data loading,
feature extraction, validation, and checkpoint saving.

See [the project plan](docs/project-plan.md) for scope, evaluation targets, and
milestones.
