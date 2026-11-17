# Background

In mouse CD8+ T cells responding to an acute viral infection, the arginine methyltransferase PRMT7
restrains clonal expansion. ATAC-seq at day 8 shows that Prmt7-deficient (Prmt7-KO) cells carry
newly accessible chromatin at the Tcf7 locus and closed chromatin at the Gzmb locus, while
accessibility at the Klrg1 locus is the same in both genotypes. A larger fraction of Prmt7-KO cells
is in cycle at day 8 than of wild-type cells. CD44 is uniformly high on antigen-experienced cells of
both genotypes and does not separate them in any assay reported here, and per-cell interferon-gamma
production is the same in the two genotypes.

# Question

At day 8 of infection, how does each marker below change in Prmt7-KO CD8+ T cells relative to
wild-type cells?

# Answer format

Reply with exactly one JSON object in this schema; it is saved verbatim to /app/output/answer.json and graded, including the rationale (at most 250 words; text beyond the cap is ignored).

```json
{
  "answer": {
    "markers": {
      "TCF1": "up|down|unchanged",
      "Granzyme B": "up|down|unchanged",
      "Ki67": "up|down|unchanged",
      "KLRG1": "up|down|unchanged",
      "CD44": "up|down|unchanged",
      "IFN-gamma": "up|down|unchanged"
    }
  },
  "rationale": "<your derivation>"
}
```
