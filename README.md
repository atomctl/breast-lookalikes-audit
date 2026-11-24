# Pathology look-alikes

For the benign breast duct look-alikes UDH and ADH (usual and atypical ductal hyperplasia), three numbers sum up the result. On one slide, a frozen-feature model ranks ADH above UDH in **78%** of region pairs more than 3 mm apart, but in only **57%** of pairs within 0.5 mm — 72% once every patient counts equally, close to chance on the pair count. Experts disagree here too: breast pathologists re-reading BRACS reproduced the ADH label with an F1 near **0.38**, against **0.55** in this cohort but **0.31** on untouched validation patients (a different measure from the pair rankings). The gap between the far and close shares is not itself established at this sample size: the registered near-versus-far contrast carries a paired 95% patient-bootstrap range of **−0.26 to +0.01**. Nearby regions also look alike in general, so the close-pair drop may reflect nearness rather than the UDH–ADH border, and nothing tried on H&E alone (75 ledger entries, most with the rule written before the result) separated touching pairs reliably.

**Not a diagnostic tool.** This is a solo research and portfolio project.

## What was built

A linear classifier used frozen UNI features from BRACS H&E regions. It combined an average of tile embeddings with a whole-region embedding. The official region split kept each patient in a single partition. An “act or ask” rule used confidence across a patient’s regions to choose between a single prediction and a shortlist for review.

The gray-zone experiments covered UDH, flat epithelial atypia (FEA), ADH, and ductal carcinoma in situ (DCIS). They tested lesion extent, tile arrangement, and duct and nuclear measurements. A separate H&E model used basal-keratin transcripts measured by Xenium as a proxy for immunostaining. Language models labelled HISTAI reports for immunostains performed or recommended.

## Key results

**The classifier plateaued.** In patient-grouped cross-validation, the frozen-feature classifier reached gray-zone macro F1 **0.705** and ADH F1 **0.55**. On the validation patients, ADH F1 was **0.31**. Pruning or softening suspect labels did not help. Fine-tuning UNI with a region-level loss reached **0.692** macro F1 and **0.554** ADH F1 across the same folds, with no clear gain. In a separate published BRACS read, three breast pathologists reproduced the ADH label with an F1 of 36.00, 35.37 and 42.29, mean **0.38** — below the 0.31 this model reaches on its own untouched validation patients, so on that measure the experts are the better reader there. That study used a different BRACS test release and the values sit in [`data/expert_f1.json`](data/expert_f1.json). [Ledger entries 56–57](LEDGER.md) · [Pati et al., HACT-Net](https://arxiv.org/abs/2102.11057), table VIII

**Errors followed slide origin.** In the out-of-fold UDH–ADH comparison, the classifier ranked UDH higher for **79** regions labelled ADH and ranked ADH higher for **67** regions labelled UDH. These counts compare the two scores directly. ADH ranked below UDH in 43 of 84 regions from ADH-labelled slides and 36 of 303 from slides labelled DCIS or invasive carcinoma. UDH ranked below ADH in 9 of 131 regions from UDH-labelled slides, 14 of 95 from ADH slides, 25 of 80 from DCIS slides, 18 of 75 from invasive carcinoma slides, and 1 of 8 from FEA slides. In the same training cohort, ADH recall was 25% on ADH-labelled slides and 66% beside carcinoma. The validation cohort did not confirm that recall gap. [Ledger entries 47 and 53](LEDGER.md)

**The gene-taught model picked up a duct-level signal and failed on the disputed cases.** On BRACS training-cohort regions, the model ranked the median predicted stain scores from highest to lowest as **UDH → ADH → FEA → DCIS**. ADH and FEA were close. UDH versus ADH AUC was **0.792**, fell to **0.60** on validation, and was **0.442** on the 146 pairwise errors above. Adding the score to the classifier gave no gain. The cell-centred retry stopped at **0.644** AUC in a held-out Xenium patient, below the **0.65** stop threshold. The duct score was never validated against matched immunostains. [Ledger entries 48–52](LEDGER.md)

**Close-together pairs are the hardest.** On one slide, the engine ranked ADH above UDH in **78%** of pairs more than 3 mm apart (1,304 pairs, 22 patients; 95% patient-bootstrap range 0.72–0.92) and in **57%** of pairs within 0.5 mm (162 pairs, 17 patients; 0.50–0.66). Weighting every patient equally moves the close figure to **72%**, and on the registered 1 mm cut the patient mean is **0.75**, so how close the close number sits to chance depends on whether pairs or patients carry the count. The far figure restricted to the 17 close-pair patients is 0.777. Accuracy rose with distance, but only pair-weighted: counting patients equally the slope is +0.02 and its range touches 0. The closest bin alone, 0–0.25 mm, sits at 54% (0.45–0.66). 147 of the 162 close pairs are training-patient pairs and 89 of those come from two patients. Regions of the same type also looked more alike when close, at the same rate, so the drop may reflect nearness in general rather than the UDH–ADH border. The steepest drop was in close pairs where both regions are plain rectangles, which can take in a neighbour's tissue; with traced outlines the drop is not shown (D = +0.14, range −0.005 to +0.28), so box overlap stays an open explanation. Pair-trained heads did not beat the engine on close pairs (entry 68 is a post hoc reading). With 17–18 patients, the close-pair result cannot separate a small signal from none. At a second hospital, Hernandez et al. found UDH-versus-ADH ranking reversed (slide-level AUC 0.21–0.41 on 19 slides); here it falls toward chance instead. The three numbers are not independent: same engine and same pair set, two windows of one distance axis, plus one literature value on a different measure. [Ledger entries 61–71](LEDGER.md) · [Hernandez et al., 2026](https://arxiv.org/abs/2611.19630)

## The wall

These routes needed no new human annotations. Each stopped before it could help resolve the disputed cases.

| Route | Why it stopped |
| --- | --- |
| Automated nuclear measurements (46) | **0 of 4** prespecified features met the criterion. AUC on the classifier’s pairwise errors was **0.49**. |
| HISTAI stain-order study gate (55) | The numerical checks passed. Visual review found **15 of 50** highest-doubt windows contained no ducts. The gate failed and the HISTAI endpoint was never scored. |
| Match HISTAI H&E to immunostains (58) | **At most 2 of 12** cases had a possible outline match, against a bar of **6**. Block IDs were unavailable. |
| BaSISS mutation maps (59) | **0** ADH regions. The probes targeted tumour mutations, and the checked release offered no aligned H&E. |
| Link HISTAI cases to later specimens (60) | **0** linkable pairs against a bar of **30**. Patient IDs and usable specimen links were absent. |

## How the work was run

The ledger puts each experiment’s rule before its result. Some analyses were exploratory, and some rules were set after an initial look. Entry 58 set its cutoff after previews of 2 of the 12 cases. The visual gate in entry 55 had no numerical cutoff. Many experiments reused the same BRACS training patients. Claude, GPT, and DeepSeek helped design and review the work. No pathologist checked any labels as part of this project.

## Limits

These findings apply to the methods and datasets tested here. Whether H&E can separate UDH from ADH under a different study design remains open. The slide pattern could reflect tissue differences or how the original labels were assigned. BACH groups both lesions under benign, so its external check cannot test this distinction.

## What would break the wall

A hospital cohort could link the original H&E biopsy, immunostains from the same block, and the corresponding excision findings or later follow-up. Matching the same ducts across sections would let us check the stain score on cases where the classifier and the label disagree. Linking follow-up to the original case would let us test whether those disagreements track what was found later.

## Data and licences

Dataset files are excluded. Download the datasets separately under their own terms.

- BRACS for non-commercial research. [Data card](steps/06-data-card.md)
- BACH under CC BY-NC-ND 4.0. [Dataset page](https://iciar2018-challenge.grand-challenge.org/Dataset/)
- HISTAI under CC BY-NC 4.0, with gated access. [HISTAI-breast](https://huggingface.co/datasets/histai/HISTAI-breast) · [HISTAI-mixed](https://huggingface.co/datasets/histai/HISTAI-mixed)
- 10x Xenium under CC BY. [Dataset catalogue](https://www.10xgenomics.com/datasets)

Code is licensed under [MIT](LICENSE).

## Reproduce

`requirements.txt` pins the environment and was frozen with `uv pip freeze` (this venv has no pip). The headline numbers need numpy alone and four small artifacts: the region table, the two out-of-fold probability files and the within-slide positions.

    python reproduce.py

It prints the far-pair share, the close-pair share and the literature F1 side by side with what this file claims, and says `match` or `MISMATCH` on each. `outputs/folds.json` holds the frozen patient folds, so the tools in `tools/` load the five folds instead of re-deriving them.

## Repo map

- [LEDGER.md](LEDGER.md) records experiment rules, results, and decisions.
- [steps/](steps/) documents methods and intermediate findings.
- [tools/](tools/) contains the experiment scripts.
