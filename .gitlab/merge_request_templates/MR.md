# 📌 Merge Request Subject

<!-- Summarize the purpose of this MR in one clear sentence -->

Title

## New Features

High-level developments

<!-- List the added features -->

*

## Fixes

List of fixes:

<!-- List the corrections / adjustments -->

*
*

# 🚶‍♂️ Walkthrough

<!-- Explain the main steps / technical impacts -->

This MR introduces ...

Additions:

<!-- List the additions -->

*

Modifications:

<!-- List the modifications -->

*

# 🗂️ Changes

| File Path          | Change Summary                |
| ------------------ | ----------------------------- |
| `path/to/file.sql` | Concise summary of the change |
| `path/to/file.sql` | Concise summary of the change |
| `path/to/file.sql` | Concise summary of the change |
| `path/to/file.sql` | Concise summary of the change |
| `path/to/file.sql` | Concise summary of the change |

# 📈 Sequence Diagram(s)

```mermaid
flowchart TD
    subgraph Ingestion and Preparation
        A[Source A] --> B[Transformation B]
        A --> C[Transformation C]
        A --> D[Transformation D]
        A --> E[Transformation E]
        A --> F[Transformation F]
        A --> G[Transformation G]
        A --> H[Transformation H]
    end

    subgraph Scoring
        B & C & D & E & F & G & H --> I[Aggregate / fact_score]
        I --> J[Historization / fact_score_histo]
        J --> K[Reporting / fact_score_ag]
    end
```
