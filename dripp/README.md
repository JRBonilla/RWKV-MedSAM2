# DRIPP - Dataset Regex Indexing & Preprocessing Pipeline

DRIPP helps index medical segmentation datasets, validate grouping regexes, and
preprocess 2D images, 3D volumes, and video frames into a consistent output
layout.

## Install

From this repository:

```bash
pip install -e dripp
```

This installs the `dripp` command.

## Create A Local Config

DRIPP ships with defaults, then optionally loads `DRIPP_CONFIG`, then a local
`dripp.ini` in the current directory.

```bash
dripp config init
```

Edit `dripp.ini` so these paths match your machine:

```ini
[paths]
base_unproc = /path/to/raw/datasets
base_proc = /path/to/preprocessed/output
index_dir = /path/to/dataset/indexes
groups_dir = /path/to/dataset/indexes/Groups
csv_filename = datasets.csv
```

Your datasets CSV is expected at:

```text
{base_unproc}/{csv_filename}
```

You can inspect the active configuration with:

```bash
dripp config show
```

## List Datasets

```bash
dripp datasets list
```

This reads the configured `datasets.csv` and prints each dataset name, modality,
and preprocessing status.

## Datasets CSV

DRIPP reads dataset metadata from:

```text
{base_unproc}/{csv_filename}
```

By default, that is `datasets.csv` under `base_unproc`. Each row describes one
dataset. DRIPP expects the raw files for a dataset to live under:

```text
{base_unproc}/{Dataset Name}/{Root Folder}
```

`Root Folder` can be blank, in which case DRIPP uses `{base_unproc}/{Dataset Name}`.

### Minimal Example

The same row is shown as a table first so it is easier to read:

| Column | Example value |
| --- | --- |
| `Dataset Name` | `ExampleCT` |
| `Modality` | `ct` |
| `Image File Type` | `.nii.gz` |
| `Mask File Type` | `.nii.gz` |
| `Root Folder` | blank |
| `Train Folders` | `imagesTr` |
| `Test Folders` | `imagesTs` |
| `Mask Folders` | `labelsTr` |
| `Mask Key` | blank |
| `Grouping Strategy` | `regex-file` |
| `Grouping Regex` | `images: (?P<id>.+)_0000\.nii\.gz; masks: (?P<id>.+)\.nii\.gz` |
| `Mask Classes` | `tumor=1\|organ=2` |
| `Segmentation Tasks` | `segmentation=tumor,organ` |
| `Background Value` | `0` |
| `Preprocessed?` | `no` |

Copy-paste CSV version:

```csv
Dataset Name,Modality,Image File Type,Mask File Type,Root Folder,Train Folders,Test Folders,Mask Folders,Mask Key,Grouping Strategy,Grouping Regex,Mask Classes,Segmentation Tasks,Background Value,Preprocessed?
ExampleCT,ct,.nii.gz,.nii.gz,,imagesTr,imagesTs,labelsTr,,regex-file,"images: (?P<id>.+)_0000\.nii\.gz; masks: (?P<id>.+)\.nii\.gz",tumor=1|organ=2,"segmentation=tumor,organ",0,no
```

### Column Reference

Required or strongly recommended columns:

| Column | Meaning |
| --- | --- |
| `Dataset Name` | Dataset identifier. It must match the dataset folder name under `base_unproc`. |
| `Modality` | Comma-separated modality names such as `ct`, `mri`, `x-ray`, `ultrasound`, `histopathology`, or `default`. |
| `Image File Type` | Comma-separated raw image extensions to index, for example `.png,.jpg` or `.nii.gz`. |
| `Mask File Type` | Comma-separated raw mask extensions to index. |
| `Root Folder` | Optional folder inside the dataset folder. Leave blank if files are directly under `{base_unproc}/{Dataset Name}`. |
| `Train Folders` | Optional comma-separated folders containing training images. If blank or unresolved, DRIPP searches the dataset directory. |
| `Test Folders` | Optional comma-separated folders containing test images. If blank, DRIPP treats indexed images as training data only. |
| `Mask Folders` | Optional comma-separated folders containing masks. If blank or unresolved, DRIPP searches the dataset directory. |
| `Mask Key` | Optional filename token used to distinguish masks from images when they share folders/extensions. |
| `Grouping Strategy` | How grouping regexes are applied: `regex-file`, `regex-folder`, `regex`, or `filename`. |
| `Grouping Regex` | Pattern spec used to pair images and masks into groups. See the regex section below. |
| `Mask Classes` | Class mapping used to label output masks. Required for indexing to produce groups. |
| `Segmentation Tasks` | Task mapping such as `task_name=class1,class2`. Required for indexing task metadata. |
| `Background Value` | Optional integer background label. Defaults to `0`. |
| `Preprocessed?` | Optional status flag. `yes` is treated as preprocessed; anything else is treated as not preprocessed. |

Folder columns support direct relative paths and simple wildcard searches. A
folder value beginning with `*` searches for directories whose path contains the
remaining text.

## Grouping Regex Patterns

Grouping regexes tell DRIPP which image files and mask files belong together.
Every regex should capture the shared case/group identifier in a named group:

```regex
(?P<id>...)
```

Any additional named groups are preserved as extra identifiers. A special
additional group named `id2` is included in output filenames and can help match
multi-image groups.

### Grouping Strategies

`Grouping Strategy` controls what string DRIPP applies each regex to:

| Strategy | Regex Target | Typical Use |
| --- | --- | --- |
| `regex-file` | The filename only, such as `case001_0000.nii.gz`. | Most datasets with matching image/mask filenames. |
| `regex-folder` | The relative path from the dataset base directory. | Datasets where folder names carry the case ID. |
| `regex` | The relative path from the dataset base directory. | Backward-compatible default; useful for full relative-path matching. |
| `filename` | No regex required; uses the base filename without extension. | Simple datasets where image and mask stems already match. |

DRIPP uses `re.fullmatch`, so patterns must match the whole target string. Use
`.*` at the beginning or end if you intentionally want a partial-style match.

### Format 1: One Dataset Pattern

Use this when one row has one modality/subdataset:

```text
images: <train_image_regex>[, <test_image_regex>]; masks: <train_mask_regex>[, <test_mask_regex>]
```

If the test regex is omitted, DRIPP reuses the train regex.

Example for nnU-Net-like CT files:

```text
images: (?P<id>.+)_0000\.nii\.gz; masks: (?P<id>.+)\.nii\.gz
```

Example with separate train/test filename conventions:

```text
images: train_(?P<id>\d+)\.png, test_(?P<id>\d+)\.png; masks: train_(?P<id>\d+)_mask\.png, test_(?P<id>\d+)_mask\.png
```

### Format 2: Multiple Subdatasets

Use this when a dataset row contains multiple modalities, anatomical subsets, or
pipelines:

```text
[Modality: Name: Pipeline] {images: <train_image_regex>[, <test_image_regex>]; masks: <train_mask_regex>[, <test_mask_regex>]}
```

`Pipeline` is optional, but when provided it must be one of:

```text
2D, 3D, Video
```

Multiple blocks can be separated by commas:

```text
[ct: liver: 3D] {images: imagesTr/(?P<id>.+)_0000\.nii\.gz; masks: labelsTr/(?P<id>.+)\.nii\.gz},
[mri: brain: 3D] {images: MRI/(?P<id>.+)\.nii\.gz; masks: masks/(?P<id>.+)\.nii\.gz}
```

With `regex-folder`, include enough relative path structure for the full match:

```text
[x-ray: chest: 2D] {images: train/images/(?P<id>[^/]+)\.png; masks: train/masks/(?P<id>[^/]+)_mask\.png}
```

### Mask Classes And Tasks

`Mask Classes` maps output class names to matching rules. Rules can be numeric
labels, label ranges, filename keywords, palette colors, or `*` as a fallback.

Single-subdataset example:

```text
tumor=1|organ=2|other=*
```

Multi-subdataset example:

```text
[liver] {liver=1|tumor=2}, [brain] {edema=1|core=2|enhancing=3}
```

`Segmentation Tasks` groups classes into task names:

```text
liver_task=liver,tumor|brain_task=edema,core,enhancing
```

For a dataset to index cleanly, class names referenced in `Segmentation Tasks`
should also appear in `Mask Classes`.

## Index Datasets

Index one dataset:

```bash
dripp index --dataset DATASET_NAME
```

Index every dataset in the CSV:

```bash
dripp index --all
```

## Preprocess Datasets

Preprocess one dataset:

```bash
dripp preprocess --dataset DATASET_NAME
```

Preprocess every dataset:

```bash
dripp preprocess --all
```

Resume all-dataset preprocessing at a specific dataset:

```bash
dripp preprocess --all --start-at DATASET_NAME
```

Limit the number of groups per split while testing:

```bash
dripp preprocess --dataset DATASET_NAME --max-groups 2
```

Enable GPU acceleration for a run:

```bash
dripp preprocess --dataset DATASET_NAME --gpu
```

## Launch The Debugger

```bash
dripp debugger
```

The debugger provides interactive regex testing, grouping inspection,
preprocessing, and preprocessed output viewing.

## Customize Output Formats

Edit the `[output_formats]` section in `dripp.ini`.

```ini
[output_formats]
2d_image_format = .png
2d_mask_format = .png
3d_image_format = .nii.gz
3d_mask_format = .nii.gz
video_frame_format = .png
video_mask_format = .png
```

Supported 2D/video formats:

```text
.png, .jpg, .jpeg, .bmp, .tif, .tiff
```

Supported 3D formats:

```text
.nii, .nii.gz, .nrrd, .mha, .mhd
```

## Troubleshooting

If `dripp datasets list` cannot find `datasets.csv`, run:

```bash
dripp config show
```

Check `base_unproc` and `csv_filename`, then update `dripp.ini`.

If a command cannot write logs, indexes, or preprocessed files, check:

```ini
[paths]
preprocessing_log_dir = ...
base_proc = ...
index_dir = ...
groups_dir = ...
```

If an output format fails validation, use one of the supported extensions above.

## Backward Compatibility

Existing module commands remain available:

```bash
python -m dripp.indexer --dataset DATASET_NAME
python -m dripp.dataset -preprocess --dataset DATASET_NAME
```
