# Background

PRMT7 is the only type III protein arginine methyltransferase, producing monomethylarginine on its substrates. In T cells it is expressed most highly in the CD8+ compartment, and cell fractionation shows it to be predominantly cytoplasmic. Endogenous PRMT7 is recovered in anti-p65 (RelA) immunoprecipitates from T cells but not in control IgG immunoprecipitates, and cells overexpressing PRMT7 show higher cytoplasmic p65 staining than untransfected cells in the same field.

**Shared assay.** Splenic CD3+ T cells are activated with anti-CD3/anti-CD28 beads and the CD4+ and CD8+ fractions are measured by flow cytometry after 72 h, each experiment against its own matched control (littermate control-genotype cells, or vehicle/DMSO).

**Arm 1 — genetic deletion.** T cells from mice with a T-cell-specific deletion of Prmt7 lack PRMT7 protein and develop normally in the thymus. After 72 h of activation their CD8+ fraction rose from about 30% in control cultures to about 48%. These cells show higher nuclear p65 after activation, higher expression of NF-kB subunits (NF-kB1, NF-kB2, RelA, RelB) and of NF-kB target genes (iNos, ICAM, Bcl-2, Bcl-xL, CD25), and down-regulation of gene-ontology terms for programmed cell death and apoptosis. Adding the NF-kB inhibitor BAY 11-7082 for the last 6 h of activation removed the difference in the CD8+ fraction between PRMT7-deficient and control cultures. In PRMT7-deficient melanoma cells, nuclear p65 was likewise higher, and BAY 11-7082 reversed that increase.

**Arm 2 — targeted protein degradation.** MS54 is a VHL-recruiting PROTAC built from the PRMT7 binder SGC8158. It lowers PRMT7 protein with DC50 2.4 uM and Dmax 79%, leaves Prmt7 mRNA unchanged, and is blocked by the proteasome inhibitor MG132 and by the neddylation inhibitor MLN4924. In T cells, 3 uM MS54 for 72 h lowered PRMT7 protein, raised nuclear and phosphorylated p65, and raised NF-kB target-gene expression. In the activation assay the CD8+ fraction rose from about 15% in vehicle-treated cultures to about 30% with MS54.

**Arm 3 — catalytic inhibition.** SGC3027 is a cell-permeable prodrug of SGC8158 that inhibits the methyltransferase activity of PRMT7. It does not lower PRMT7 protein levels, and pre-treating cells with SGC3027 completely abolishes MS54-induced PRMT7 degradation, showing that SGC3027 occupies PRMT7 inside cells. In the same activation assay, SGC3027 did not expand the CD8+ compartment: the CD8+ fraction fell from about 30% in vehicle-treated cultures to about 11%, with a modest reciprocal rise in CD4+ cells.

# Question

State the primary mechanism by which PRMT7 restrains NF-kB-driven CD8+ T-cell expansion — one sentence that accounts for the split between the three arms above — and list the key entities or features that mechanism requires (3 to 6 short strings: molecules, activities, localisations or modalities).

# Answer format

Reply with exactly one JSON object in this schema; it is saved verbatim to /app/output/answer.json and graded, including the rationale (at most 400 words; text beyond the cap is ignored).

```json
{
  "answer": {
    "mechanism_primary": "one sentence naming the mechanism",
    "key_entities": ["short string", "short string", "short string"]
  },
  "rationale": "the evidence and reasoning behind that mechanism"
}
```
