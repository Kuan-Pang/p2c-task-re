# Background

Mouse bone-marrow-derived macrophages carry exactly one of four genetic perturbations, applied on
its own. In the perturbed cells: lipopolysaccharide stimulation produces no detectable secreted
IL-10; lipopolysaccharide stimulation produces the normal amount of secreted TNF; and adding
recombinant IL-10 protein to the cells produces the normal amount of STAT3 phosphorylation.
MyD88 is required for every lipopolysaccharide-driven cytokine in these cells. IL-10 signals through
the IL-10 receptor alpha chain and STAT3.

# Question

Which of the four perturbations was applied?

# Answer format

Reply with exactly one JSON object in this schema; it is saved verbatim to /app/output/answer.json and graded, including the rationale (at most 250 words; text beyond the cap is ignored).

```json
{
  "answer": {
    "choice": "Stat3 deletion | Il10 deletion | Il10ra deletion | Myd88 deletion"
  },
  "rationale": "<your derivation>"
}
```
