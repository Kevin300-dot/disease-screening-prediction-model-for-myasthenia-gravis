# South Chennai hospital finder - data and method

The finder is limited to South Chennai. The MG prediction is unchanged; the finder only adds a
`care_locator` block to the report after prediction.

| Hospital | Location | What the hospital's own website says (checked Oct 2026) | MG score |
|---|---|---|---|
| MIOT International | 4/112, Mount Poonamallee Road, Manapakkam, Chennai 600 089 (south-west Chennai) | Neurology page lists myasthenia gravis; neurophysiology lab with nerve conduction study and needle EMG | 6/6 |
| Prashanth Hospitals | No. 36 & 36A, Velachery Main Road, Velachery, Chennai 600 042 | Neurology team article names myasthenia gravis, neuromuscular disorders and nerve-conduction studies | 4/6 |
| Apollo Speciality Hospitals, OMR | 5/639, Rajiv Gandhi Salai, Tirumalai Nagar, Perungudi, Chennai 600 096 | Neurology listed as a special-focus specialty (MG not named) | 1/6 |
| Gleneagles Global Health City | 439, Embassy Residency Road, Cheran Nagar, Perumbakkam, Chennai 600 100 | Neurology department listed (MG not named) | 1/6 |
| Kauvery Hospital, Radial Road | No. 2/473, Radial Road, Kovilambakkam, Chennai 600 129 | Neuro Sciences centre of excellence (MG not named) | 1/6 |
| Dr. Rela Institute & Medical Centre | #7, CLC Works Road, Nagappa Nagar, Chromepet, Chennai 600 044 | Neurosciences centre and Department of Neurology (MG not named) | 1/6 |
| MGM Malar Hospital | Old No. 52, New No. 111, 1st Main Road, Gandhi Nagar, Adyar, Chennai 600 020 | Neurology and neurosurgery page (MG not named) | 1/6 |

Source URLs are stored in `backend/mg_predictor/mg/hospitals.py` (field `source`).

* **Nearest** = smallest straight-line distance from the chosen area (or GPS position).
* **Best** = highest evidence score: MG named on the hospital's own neurology page 3 (or in its neurology
  team's article 2) + EMG / nerve-conduction laboratory 2 (or nerve-conduction studies mentioned 1) +
  neurology department 1. Ties go to the nearer hospital. This is a transparent rule, not a clinical
  outcome ranking.
* Hospital pins and area centres are approximate (a few hundred metres); distances are straight-line
  estimates. The "Directions on Google Maps" link in the report gives the real route.
* To add or correct a hospital edit `HOSPITALS` in `mg/hospitals.py`. Areas live in `AREAS` and in
  `src/lib/mg-locations.ts` (a test checks they agree).
