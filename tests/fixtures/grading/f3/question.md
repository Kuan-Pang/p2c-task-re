# Background

Quiescent human dermal fibroblasts were treated for 48 hours with one of four cytokines, and
proliferation was read as EdU incorporation. In these cells NF-kB activation is the step that drives
the proliferative response, and IL-1beta and TNF-alpha activate NF-kB to the same degree at the doses
used. IL-6 does not activate NF-kB in these cells; it signals through STAT3, which raises EdU
incorporation about half as much as NF-kB activation does. TGF-beta drives the cells into a
myofibroblast state and arrests the cell cycle.

# Question

Rank the four cytokines by the size of the proliferative response they produce, strongest first,
grouping any that the premises make equal into the same tie set.

# Answer format

Reply with exactly one JSON object in this schema; it is saved verbatim to /app/output/answer.json and graded, including the rationale (at most 250 words; text beyond the cap is ignored).

```json
{
  "answer": {
    "ranking": [
      [
        "IL-6"
      ],
      [
        "IL-1beta",
        "TGF-beta"
      ],
      [
        "TNF-alpha"
      ]
    ]
  },
  "rationale": "<your derivation>"
}
```
