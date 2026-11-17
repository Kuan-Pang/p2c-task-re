# Background

Human monocyte-derived macrophages were stimulated for 6 hours with 100 ng/ml lipopolysaccharide
(LPS) and secreted TNF was measured. LPS alone raises TNF about twenty-fold over unstimulated cells;
that arm is the reference for every other arm below. LPS is recognised at the cell surface by the
TLR4-MD-2 complex, the only LPS receptor these cells express. The co-receptor CD14 concentrates LPS
and hands it to TLR4; in this preparation blocking CD14 lowers the response to 1 ng/ml LPS but has no
effect at 100 ng/ml. Polymyxin B binds the lipid A moiety of LPS and keeps it away from any receptor.
Priming with interferon-gamma for 16 hours before stimulation raises surface TLR4 and MD-2.

# Question

Taking the LPS-alone arm as the reference, what happens to secreted TNF at 6 hours in each arm
below? Use increased (above the LPS-alone arm), lost (no TNF above the unstimulated control) or
unchanged (indistinguishable from the LPS-alone arm).

# Answer format

Reply with exactly one JSON object in this schema; it is saved verbatim to /app/output/answer.json and graded, including the rationale (at most 250 words; text beyond the cap is ignored).

```json
{
  "answer": {
    "arms": {
      "LPS + anti-TLR4": "increased|lost|unchanged",
      "LPS + anti-CD14": "increased|lost|unchanged",
      "LPS + polymyxin B": "increased|lost|unchanged",
      "LPS + IFN-gamma priming": "increased|lost|unchanged"
    }
  },
  "rationale": "<your derivation>"
}
```
