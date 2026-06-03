# manifests/ — evaluation sets

The exact image/query/relevance sets the protocol runs on. Only this metadata is
bundled; the product **images** are public (Amazon ESCI) and are not included — see
"Obtaining images" below.

| File | Set |
|---|---|
| `esci500_manifest.csv` | headline set: **491 images / 500 paired triples** (125 per cohort), balanced across the four ESCI relevance classes |
| `esci1500_manifest.csv` | the **1,430-pair** scale-up (3× larger, label-balanced) |

## Columns

| Column | Meaning |
|---|---|
| `example_id` | ESCI example identifier |
| `query_id` | ESCI query identifier |
| `query` | the shopper query string (ESCI is multilingual — non-English queries are expected) |
| `product_id` | the Amazon **ASIN** |
| `product_title` | seller-side product title (the optimization anchor source for CoGEO/PGD-bare) |
| `esci_label` | relevance class: **E**xact / **S**ubstitute / **C**omplement / **I**rrelevant |
| `image_path` | relative path `img/<ASIN>.jpg` |

## Obtaining images

Images are addressed as `img/<ASIN>.jpg`. Reconstruct them from the public
Amazon ESCI dataset (`amazon-science/esci-data`) by `product_id` (ASIN), placing each
as `img/<ASIN>.jpg` relative to the manifest. The loaders in `../src/data/` build this
layout; no images are redistributed here.
