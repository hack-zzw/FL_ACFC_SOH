# Data used by the reproduction commands

`processed/core/` contains the frozen base per-record features used by the study. These feature tables follow the public processed-data representation described in the original dataset/preprocessing sources. The PINN4SOH repository explicitly provides processed data and links the raw HUST, MIT, TJU and XJTU data and preprocessing library:

- [Processed data and original-source links](https://github.com/wang-fujin/PINN4SOH)
- [Original preprocessing library](https://github.com/wang-fujin/Battery-dataset-preprocessing-code-library)
- [Associated source paper](https://doi.org/10.1038/s41467-024-48779-z)

The proposed SOH model uses ACfC. Reusing these processed observations does not import the PINN neural architecture.

`processed/domain/source/` contains seven source-side window/QV descriptor assets. `processed/domain/target/` contains the corresponding target observations. `controls/prepared/` stores the223 numeric descriptors,36-dimensional frozen embeddings, task definitions and evaluation truth. Missing observations remain missing until the registered source normalization/imputation step. Target task/observation inputs omit query SOH; `target_truth.csv` is read by the evaluation code.

`external/protocol_features/` and `external/manifests/` contain the frozen NASA/Oxford inputs and protocol metadata. Oxford development cells are2,3,5,7,8 and evaluation cells are1,4,6. NASA20/40/60-label main evaluation uses the same200 queries from10 batteries; Oxford uses111 common queries from3 evaluation batteries for0/5/10/20 labels.

## Original measurements

The providers and identifiers are listed in `original_sources.json`. HUST version2 is linked at its Mendeley record; XJTU is linked at the original Zenodo record; the original preprocessing repository links TJU and MIT. NASA and Oxford are linked at their institutional repositories. Provider records retain their own access and reuse conditions. A missing `observed_licence` value means that a redistribution licence was not established by this packaging audit; it does not mean the data have no conditions.

The raw measurement archives are not duplicated here. For the two external archives, the original-file hashes used by this study are:

| Original filename | SHA-256 |
|---|---|
| `5.+Battery+Data+Set.zip` | `82302a7db4fc1b34e0b6676326610438d43b816bdf11a69d1d012a464ef2f92e` |
| `Oxford_Battery_Degradation_Dataset_1.mat` | `a8f0b928f4ea77d7ba98b97194b78cc77336f2338fee8b57573d043f0cf26781` |

`python train.py prepare` reconstructs the controlled-experiment descriptors and embeddings from the supplied processed observations. It is not a claim that all original measurement formats are converted by a single raw-data command. The source dataloader and `configs/data/` specify which batteries/records enter the study. Main-model source training uses205,946 prefix records; the controlled readouts use236,401 post-support source records from the full242,461-record source pool.

## Fields and units

`dictionary.csv` lists supplied fields and roles. `design_schema.csv` specifies the223 numeric descriptors and36 encoder features. SOH/true_soh is a capacity ratio; reported SOH error percentage points equal100 times the ratio error. `cycle_index` is the retained observation index, and `native_cycle` is the source cycle coordinate where available. Native-current and temperature fields explicitly ending in`_A`/`_C` retain A/°C; time/gap fields ending in`_s` retain seconds. Other base features follow the source preprocessing definition; ambiguous physical units are not inferred from their names. Record keys, quality flags and missingness masks are described separately from physical observables.

Each file in this release has a path, byte count and SHA-256 in `manifests/files.json`. These identify the actual data version independently of a provider's changing default branch. Data rights are separate from the software licence.


Source-side design/embedding caches are regenerated automatically before control or ridge training. They are omitted from the minimal download; measured/processed inputs and all model-replay inputs are included. The generated cache is written under outputs/feature_cache.
