# Chennai hospital finder - data and method

Covers Chennai (50 selectable localities, or GPS inside Greater Chennai). The MG prediction is unchanged; the
finder only adds a `care_locator` block to the report after prediction.

| Hospital | Location | What the hospital's own pages say (checked Oct-Nov 2026) | MG score |
|---|---|---|---|
| MIOT International | Manapakkam | Neurology page lists myasthenia gravis; neurophysiology lab with NCS and needle EMG | 6/6 |
| Apollo Hospitals, Greams Road | Greams Road | Neurosciences page: "care for muscle diseases like myasthenia gravis"; EMG and NCS listed | 6/6 |
| Rajiv Gandhi Govt. General Hospital - Institute of Neurology | Park Town | Department listing names MG, weekly Neuromuscular Clinic, NCS/EMG | 5/6 |
| Kauvery Hospital, Alwarpet | Alwarpet / Mylapore | Neurophysiology dept lists NCS, EMG, neuromuscular clinics; press release on MG (branch not stated) | 5/6 |
| Prashanth Hospitals | Velachery | Neurology team article names MG and nerve-conduction studies | 4/6 |
| Apollo Speciality Hospitals, OMR | Perungudi | Neurology listed (MG not named) | 1/6 |
| Gleneagles Global Health City | Perumbakkam | Neurology department listed (MG not named) | 1/6 |
| Kauvery Hospital, Radial Road | Kovilambakkam | Neuro Sciences centre (MG not named) | 1/6 |
| Dr. Rela Institute & Medical Centre | Chromepet | Neurosciences centre (MG not named) | 1/6 |
| MGM Malar Hospital | Adyar | Neurology and neurosurgery (MG not named) | 1/6 |

Source URLs are in `backend/mg_predictor/mg/hospitals.py` (field `source`).

* **Nearest** = smallest straight-line distance from the chosen area (or GPS position).
* **Best** = highest evidence score: MG named on the hospital's neurology page 3 (or in its team article /
  department listing 2) + EMG/nerve-conduction lab 2 + neurology department 1. Ties go to the nearer hospital.
  A transparent rule, not a clinical outcome ranking.
* Pins and area centres are approximate (a few hundred metres). The Google Maps link gives the real route.
* A low score means "MG not named on the page", not "does not treat MG".
* To add or correct a hospital edit `HOSPITALS` in `mg/hospitals.py`; areas live in `AREAS` and in
  `src/lib/mg-locations.ts` (a test checks they agree).
