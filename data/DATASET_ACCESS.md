# ProAct Dataset Access

Access is for individually approved researchers conducting **non-commercial scientific research** under the public [Data Use Agreement](https://github.com/ZhuXMMM/ProAct/blob/6afeb9c6c54d0ef649a741a01c40e6e2c3816383/data/DATA_USE_AGREEMENT.md). Read it before accepting the application form. The dataset has separate terms from the repository's Apache 2.0-licensed code.

The release contains 495 masked, silent 3840×2160 self-collected Tencent videos and 18,747 masked 768×432 training keyframes. It covers 165 takes and 15 annotated tasks, with 569 annotation records and 2,918 step segments. Existing splits contain 324 train and 171 test videos; no validation split is added. The 75-task catalog is `reference_only`.

## Apply with your own Hugging Face account

1. Read the public [Data Use Agreement](https://github.com/ZhuXMMM/ProAct/blob/6afeb9c6c54d0ef649a741a01c40e6e2c3816383/data/DATA_USE_AGREEMENT.md), then sign in to your own individual Hugging Face account.
2. Open the [ProAct dataset page](https://huggingface.co/datasets/Zhuxmmm/ProAct) in your browser and complete its research-access form.
3. Provide your **full name, institution or organization, institutional email, role or position, and research purpose** (research question, planned use and expected outputs). Accept each of the three statements below.
4. The application sends a real-time notification to zhuxiaomeng224@gmail.com. The maintainer reviews it and grants access to your individual account if approved. Submission does not automatically grant access; review time varies.

Each researcher needs separate approval, including members of the same lab or project. Keep your account and access token private; approval cannot be shared with colleagues.

## Applicant acceptance

- [ ] I will use the ProAct dataset only for non-commercial scientific research as described in my application, and will obtain prior written permission for products, commercial services, or other commercial exploitation.
- [ ] I will not identify or attempt to identify people in the dataset, link the data to identifying information, contact people depicted in it, or reverse or attempt to reverse privacy transformations.
- [ ] I will not redistribute or mirror the dataset, expose restricted content in derived material, bypass access controls, or share credentials or access tokens without prior written permission; each researcher must apply separately.

## Download after approval

Use the approved account with the [official Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli):

```sh
hf auth login
hf download Zhuxmmm/ProAct --repo-type dataset --local-dir ProAct
```

## Data scope and limitations

Original training filenames are retained, while `metadata/keyframes.jsonl` records each image's actual decoded ordinal in the final masked native video. There are 12,727 strong numeric alignments and 6,020 disclosed historical inferences. The 5,684 preceding-frame selections include the six historical endpoint labels. Privacy processing and recompression change pixels; retained filenames do not imply pixel identity with old training JPEGs.

Source annotation label discrepancies and embedded camera-ID conflicts are preserved and flagged in the metadata. COIN, Ego-Exo4D and UCF-Crime media are acquired separately under their own terms; see the [data overview](README.md#external-source-media) for official starting points.

Send access questions, requests for written exceptions and private reports of privacy concerns to [zhuxiaomeng224@gmail.com](mailto:zhuxiaomeng224@gmail.com).
