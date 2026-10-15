# ProAct Data

The self-collected research release is hosted on [Hugging Face](https://huggingface.co/datasets/Zhuxmmm/ProAct) with individual manual approval for **non-commercial scientific research**. This directory contains the public [application instructions](DATASET_ACCESS.md) and [Data Use Agreement](https://github.com/ZhuXMMM/ProAct/blob/6afeb9c6c54d0ef649a741a01c40e6e2c3816383/data/DATA_USE_AGREEMENT.md); media are downloaded separately after approval.

| Item | Included |
| --- | --- |
| Masked videos | 495 at 3840×2160, with no audio track |
| Masked training keyframes | 18,747 JPEGs at 768×432 |
| Takes / annotated tasks | 165 / 15 |
| Annotation records / step segments | 569 / 2,918 |
| Existing video split | 324 train / 171 test; no validation split |
| Full 75-task graph catalog | `reference_only`; no additional media coverage |

## Access

Read the agreement, sign in to your own Hugging Face account, and complete the dataset page's research-access form. Applications notify zhuxiaomeng224@gmail.com in real time and are reviewed manually. Each researcher applies separately; submission does not automatically grant access.

After approval, use the approved account:

```sh
hf auth login
hf download Zhuxmmm/ProAct --repo-type dataset --local-dir ProAct
```

See [Dataset Access](DATASET_ACCESS.md) for the required fields and acceptance statements. The code's Apache 2.0 license does not cover these data.

## Frame labels and research limitations

Original training filenames are retained. Each JPEG is re-extracted from the final masked native video at the actual decoded ordinal recorded in `metadata/keyframes.jsonl`, then resized to 768×432. Use `frame_index_zero_based` for the actual ordinal and `filename_frame_label` for the historical filename label; they can differ.

Of the 18,747 alignments, **12,727 have strong numeric matches** and **6,020 use disclosed historical inference**. There are **5,684 preceding-frame selections**, including the six historical endpoint labels that select the last available frame N−1. Those six are included in 5,684. Privacy masks and recompression change pixels; filename preservation does not imply pixel identity with the historical JPEGs.

Source task-graph/vocabulary label discrepancies and conflicting embedded camera IDs are preserved and flagged in `metadata/validation_notes.json`. Original labels, timestamps and split assignments are retained. Historical keyframe alignments marked as inference remain uncertain and should be considered when selecting examples or evaluating results.

## External source media

COIN, Ego-Exo4D and UCF-Crime media are not reuploaded here. Obtain them through their official sources and follow each source's terms; ProAct approval does not grant access to them.

| Source | Official access and download instructions |
| --- | --- |
| COIN | [Dataset website](https://coin-dataset.github.io/) |
| Ego-Exo4D | [Getting Started](https://docs.ego-exo4d-data.org/getting-started/) |
| UCF-Crime | [Official project page](https://www.crcv.ucf.edu/research/real-world-anomaly-detection-in-surveillance-videos/) |

Contact: [zhuxiaomeng224@gmail.com](mailto:zhuxiaomeng224@gmail.com).
