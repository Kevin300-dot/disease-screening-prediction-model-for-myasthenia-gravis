"""Chennai care locator: nearest hospital and best hospital for myasthenia gravis (MG).

This module is completely separate from the prediction models - it never touches the MG
screening result. It only looks at where the patient is and which real hospitals in
Chennai treat neurological disease.

DATA PROVENANCE
  * Every hospital below is a real hospital. Names, addresses and the "evidence" text were read
    from the hospital's own website (URL kept in `source`) in October 2026.
  * `lat`/`lng` are approximate map pins (within a few hundred metres). They are used only for
    straight-line distance estimates; the Google Maps link in each result gives the true route.
  * Area centres in AREAS are approximate centre points of each locality.

HOW "BEST" IS DECIDED (transparent, rule-based - it is NOT a clinical outcome ranking)
  evidence score =  3  hospital's own neurology/department page names myasthenia gravis
                    2  ... or its neurology team's article / department listing names myasthenia gravis
                    2  hospital lists an EMG / nerve-conduction laboratory
                    1  ... or its team's article mentions nerve-conduction studies
                    1  has a neurology department / neurosciences centre
  The highest score is the "best" hospital for MG. Ties are broken by the shorter distance.
  "Nearest" is simply the smallest straight-line distance.
"""
from __future__ import annotations

import math
from urllib.parse import quote_plus

REGION = "Chennai"
DISCLAIMER = (
    "Hospital information was taken from each hospital's own website and may change. Distances are "
    "approximate straight-line estimates from the centre of your area; open the map link for the real "
    "route. 'Best' means the strongest published evidence of myasthenia gravis care, not a clinical "
    "ranking. Always phone the hospital to confirm that a neurologist is available before travelling."
)

# Bounding box that counts as "Chennai" for GPS locations (Greater Chennai: Thiruvottiyur / Ennore in the
# north to Perumbakkam / Tambaram in the south, Avadi / Porur in the west to the coast in the east).
BOUNDS = {"lat_min": 12.80, "lat_max": 13.25, "lng_min": 80.05, "lng_max": 80.35}

# Approximate centre points of Chennai localities: key -> (label, lat, lng)
AREAS: dict[str, tuple[str, float, float]] = {
    "saidapet": ("Saidapet", 13.0213, 80.2231),
    "guindy": ("Guindy", 13.0067, 80.2206),
    "kotturpuram": ("Kotturpuram", 13.0129, 80.2419),
    "adyar": ("Adyar", 13.0012, 80.2565),
    "besant_nagar": ("Besant Nagar", 13.0002, 80.2668),
    "thiruvanmiyur": ("Thiruvanmiyur", 12.9830, 80.2594),
    "taramani": ("Taramani", 12.9863, 80.2415),
    "velachery": ("Velachery", 12.9815, 80.2180),
    "perungudi": ("Perungudi", 12.9650, 80.2461),
    "thoraipakkam": ("Thoraipakkam", 12.9416, 80.2363),
    "sholinganallur": ("Sholinganallur", 12.9010, 80.2279),
    "pallikaranai": ("Pallikaranai", 12.9340, 80.2140),
    "perumbakkam": ("Perumbakkam", 12.9005, 80.2070),
    "medavakkam": ("Medavakkam", 12.9171, 80.1923),
    "kovilambakkam": ("Kovilambakkam", 12.9368, 80.1927),
    "madipakkam": ("Madipakkam", 12.9624, 80.1986),
    "nanganallur": ("Nanganallur", 12.9815, 80.1895),
    "alandur": ("Alandur", 13.0030, 80.2020),
    "pallavaram": ("Pallavaram", 12.9675, 80.1491),
    "chromepet": ("Chromepet", 12.9516, 80.1462),
    "tambaram": ("Tambaram", 12.9249, 80.1000),
    "selaiyur": ("Selaiyur", 12.9100, 80.1400),
    # central / north / west Chennai
    "anna_nagar": ("Anna Nagar", 13.0850, 80.2101),
    "alwarpet": ("Alwarpet", 13.0339, 80.2540),
    "ambattur": ("Ambattur", 13.1143, 80.1548),
    "ashok_nagar": ("Ashok Nagar", 13.0373, 80.2123),
    "avadi": ("Avadi", 13.1067, 80.1090),
    "egmore": ("Egmore", 13.0732, 80.2609),
    "ekkaduthangal": ("Ekkaduthangal", 13.0170, 80.2050),
    "george_town": ("George Town", 13.0950, 80.2870),
    "kilpauk": ("Kilpauk", 13.0827, 80.2420),
    "kk_nagar": ("K.K. Nagar", 13.0415, 80.1996),
    "kodambakkam": ("Kodambakkam", 13.0520, 80.2250),
    "kolathur": ("Kolathur", 13.1210, 80.2160),
    "manapakkam": ("Manapakkam", 13.0245, 80.1865),
    "mogappair": ("Mogappair", 13.0850, 80.1700),
    "mylapore": ("Mylapore", 13.0368, 80.2676),
    "nungambakkam": ("Nungambakkam", 13.0569, 80.2425),
    "park_town": ("Park Town", 13.0830, 80.2750),
    "perambur": ("Perambur", 13.1186, 80.2377),
    "porur": ("Porur", 13.0382, 80.1565),
    "royapettah": ("Royapettah", 13.0540, 80.2640),
    "royapuram": ("Royapuram", 13.1130, 80.2950),
    "t_nagar": ("T. Nagar", 13.0418, 80.2341),
    "thiruvottiyur": ("Thiruvottiyur", 13.1640, 80.3000),
    "tondiarpet": ("Tondiarpet", 13.1259, 80.2870),
    "triplicane": ("Triplicane", 13.0580, 80.2750),
    "vadapalani": ("Vadapalani", 13.0500, 80.2121),
    "valasaravakkam": ("Valasaravakkam", 13.0420, 80.1760),
    "villivakkam": ("Villivakkam", 13.1080, 80.2060),
}

HOSPITALS: list[dict] = [
    {
        "id": "miot",
        "name": "MIOT International",
        "area": "Manapakkam (west Chennai)",
        "address": "4/112, Mount Poonamallee Road, Manapakkam, Chennai - 600 089",
        "lat": 13.0245, "lng": 80.1865,
        "phone": None,
        "department": "Neurology and Neurosciences",
        "evidence": [
            "Its neurology page lists myasthenia gravis (an autoimmune neuromuscular disease) among the conditions it handles in emergencies.",
            "Neurophysiology laboratory offers nerve conduction study of all four limbs and needle EMG.",
        ],
        "points": {"mg_named_on_hospital_page": 3, "emg_ncs_lab": 2, "neurology_department": 1},
        "source": "https://www.miotinternational.com/key-specialities/neurology-and-neurosciences/",
    },
    {
        "id": "prashanth",
        "name": "Prashanth Hospitals",
        "area": "Velachery",
        "address": "No. 36 & 36A, Velachery Main Road, Velachery, Chennai - 600 042",
        "lat": 12.9788, "lng": 80.2186,
        "phone": "044 4680 5544",
        "department": "Neurology and Neurosurgery",
        "evidence": [
            "Its neurology team's own article states that its neurologists manage conditions such as myasthenia gravis and has a section on neuromuscular and autoimmune disorders.",
            "The same article mentions nerve-conduction studies.",
            "Velachery branch page lists Neurology & Neurosurgery among its specialties.",
        ],
        "points": {"mg_named_in_team_article": 2, "ncs_mentioned": 1, "neurology_department": 1},
        "source": "https://prashanthhospitals.org/blogs/neurology/best-neurologist-in-chennai/",
    },
    {
        "id": "apollo_omr",
        "name": "Apollo Speciality Hospitals, OMR",
        "area": "Perungudi",
        "address": "5/639, Rajiv Gandhi Salai (OMR), Tirumalai Nagar, Perungudi, Chennai - 600 096",
        "lat": 12.9676, "lng": 80.2468,
        "phone": None,
        "department": "Neurology",
        "evidence": ["Hospital page lists Neurology as a special-focus specialty (disorders of the brain, spinal cord and nervous system). Myasthenia gravis is not named on the page."],
        "points": {"neurology_department": 1},
        "source": "https://apollohospitals.com/hospitals/apollo-speciality-hospitals-omr-chennai",
    },
    {
        "id": "gleneagles",
        "name": "Gleneagles Global Health City",
        "area": "Perumbakkam",
        "address": "439, Embassy Residency Road, Cheran Nagar, Perumbakkam, Chennai - 600 100",
        "lat": 12.8985, "lng": 80.2016,
        "phone": "+91 92402 62425 (appointments)",
        "department": "Neurology",
        "evidence": ["Hospital lists a Neurology department with a team of senior neurology consultants. Myasthenia gravis is not named on the page."],
        "points": {"neurology_department": 1},
        "source": "https://www.fortishealthcare.com/location/gleneagles-hospital-chennai",
    },
    {
        "id": "kauvery_radial",
        "name": "Kauvery Hospital, Radial Road",
        "area": "Kovilambakkam",
        "address": "No. 2/473, Radial Road, Kovilambakkam, Chennai - 600 129",
        "lat": 12.9355, "lng": 80.1924,
        "phone": None,
        "department": "Neuro Sciences (Neurology and Neurosurgery)",
        "evidence": ["Neuro Sciences is listed as a Centre of Excellence at this branch, with Neurology and Neurosurgery appointments. Myasthenia gravis is not named on the page."],
        "points": {"neurology_department": 1},
        "source": "https://www.kauveryhospital.com/?p=8671",
    },
    {
        "id": "rela",
        "name": "Dr. Rela Institute & Medical Centre",
        "area": "Chromepet",
        "address": "#7, CLC Works Road, Nagappa Nagar, Chromepet, Chennai - 600 044",
        "lat": 12.9536, "lng": 80.1386,
        "phone": None,
        "department": "Department of Neurology / Institute of Neuro Sciences",
        "evidence": ["Neurosciences is a Centre of Excellence with a Department of Neurology. Myasthenia gravis is not named on the page."],
        "points": {"neurology_department": 1},
        "source": "https://www.relainstitute.com/locations",
    },
    {
        "id": "mgm_malar",
        "name": "MGM Malar Hospital",
        "area": "Adyar",
        "address": "Old No. 52, New No. 111, 1st Main Road, Gandhi Nagar, Adyar, Chennai - 600 020",
        "lat": 13.0058, "lng": 80.2538,
        "phone": None,
        "department": "Neurology and Neurosurgery",
        "evidence": ["Neurology page covers brain, spine and nerve disorders including epilepsy and neurodegenerative disease. Myasthenia gravis is not named on the page."],
        "points": {"neurology_department": 1},
        "source": "https://mgmmalar.in/speciality/neurology/",
    },
    {
        "id": "rgggh",
        "name": "Rajiv Gandhi Government General Hospital - Institute of Neurology",
        "area": "Park Town (central Chennai)",
        "address": "Poonamallee High Road, Park Town, Chennai - 600 003",
        "lat": 13.0814, "lng": 80.2756,
        "phone": None,
        "department": "Institute of Neurology (Madras Medical College)",
        "evidence": [
            "The Institute's department listing names myasthenia gravis under neuromuscular diseases and runs a weekly Neuromuscular Clinic.",
            "Diagnostic facilities listed include nerve conduction study (NCS) and electromyography (EMG).",
            "Government teaching hospital. Check its outpatient timings before you travel.",
        ],
        "points": {"mg_named_on_department_listing": 2, "emg_ncs_lab": 2, "neurology_department": 1},
        "source": "https://mmc.xenovex.com/en/education/madras-medical-college/department/super-speciality/neurology",
    },
    {
        "id": "apollo_greams",
        "name": "Apollo Hospitals, Greams Road",
        "area": "Greams Road (central Chennai)",
        "address": "21, Greams Lane, Off Greams Road, Chennai - 600 006",
        "lat": 13.0627, "lng": 80.2517,
        "phone": None,
        "department": "Apollo Institute of Neurosciences",
        "evidence": [
            "Its neurosciences page states: 'We provide care for muscle diseases like myasthenia gravis.'",
            "Diagnostics listed on that page include electromyography (EMG) and nerve conduction studies.",
        ],
        "points": {"mg_named_on_hospital_page": 3, "emg_ncs_lab": 2, "neurology_department": 1},
        "source": "https://www.apollohospitals.com/chennai/clinical-excellence/neurosciences",
    },
    {
        "id": "kauvery_alwarpet",
        "name": "Kauvery Hospital, Alwarpet",
        "area": "Alwarpet / Mylapore",
        "address": "199, Luz Church Road, Mylapore, Chennai - 600 004",
        "lat": 13.0390, "lng": 80.2540,
        "phone": None,
        "department": "Neurology and Neurophysiology",
        "evidence": [
            "Neurology and Neurophysiology department lists nerve conduction studies, EMG and neuromuscular clinics.",
            "Kauvery's own press release (Aug 2022) describes treating two patients with myasthenia gravis; it names 'Kauvery Hospital Chennai' without a branch, so confirm the branch by phone.",
        ],
        "points": {"mg_named_in_team_article": 2, "emg_ncs_lab": 2, "neurology_department": 1},
        "source": "https://www.kauveryhospital.com/?p=11618",
    },
]


def _km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def area_options() -> list[dict]:
    return [{"key": k, "label": v[0]} for k, v in sorted(AREAS.items(), key=lambda kv: kv[1][0])]


def evidence_score(h: dict) -> int:
    return sum(h["points"].values())


def resolve_origin(location) -> dict:
    """Turn the request's `location` into {label, lat, lng, source}. Raises ValueError when the
    location is unknown or outside Chennai."""
    if not isinstance(location, dict):
        raise ValueError("location must be an object such as {\"area\": \"velachery\"}")
    if location.get("area"):
        key = str(location["area"]).strip().lower().replace(" ", "_")
        if key not in AREAS:
            raise ValueError(
                f"Unknown area '{location['area']}'. MyaScreen's hospital finder covers {REGION}.")
        label, lat, lng = AREAS[key]
        return {"label": label, "lat": lat, "lng": lng, "source": "area"}
    if "lat" in location and "lng" in location:
        try:
            lat, lng = float(location["lat"]), float(location["lng"])
        except (TypeError, ValueError):
            raise ValueError("lat and lng must be numbers") from None
        b = BOUNDS
        if not (b["lat_min"] <= lat <= b["lat_max"] and b["lng_min"] <= lng <= b["lng_max"]):
            raise ValueError(
                f"That location is outside {REGION}. MyaScreen's hospital finder covers {REGION} only - "
                f"please pick your area from the list.")
        return {"label": "Your current location", "lat": round(lat, 5), "lng": round(lng, 5), "source": "gps"}
    raise ValueError("Provide either an area or lat/lng in location")


def _maps_url(origin: dict, h: dict) -> str:
    dest = quote_plus(f"{h['name']}, {h['address']}")
    return (f"https://www.google.com/maps/dir/?api=1&origin={origin['lat']},{origin['lng']}"
            f"&destination={dest}&travelmode=driving")


def _public(origin: dict, h: dict, dist: float) -> dict:
    return {
        "id": h["id"], "name": h["name"], "area": h["area"], "address": h["address"],
        "phone": h["phone"], "department": h["department"], "evidence": h["evidence"],
        "evidence_score": evidence_score(h), "max_score": 6,
        "mg_named": any(k.startswith("mg_named") for k in h["points"]),
        "distance_km": round(dist, 1), "maps_url": _maps_url(origin, h), "source": h["source"],
    }


def locate(location) -> dict:
    """Nearest and best MG-capable hospital in Chennai for the given location."""
    origin = resolve_origin(location)
    rows = [(h, _km(origin["lat"], origin["lng"], h["lat"], h["lng"])) for h in HOSPITALS]
    nearest_h, nearest_d = min(rows, key=lambda r: r[1])
    best_h, best_d = min(rows, key=lambda r: (-evidence_score(r[0]), r[1]))
    ranking = [_public(origin, h, d) for h, d in sorted(rows, key=lambda r: (-evidence_score(r[0]), r[1]))]
    same = nearest_h["id"] == best_h["id"]
    if same:
        summary = f"{nearest_h['name']} is both the nearest and the best-evidenced hospital for MG care from {origin['label']}."
    else:
        summary = (f"Nearest: {nearest_h['name']} ({nearest_d:.1f} km). Best for MG: {best_h['name']} "
                   f"({best_d:.1f} km), chosen for the strongest published evidence of myasthenia gravis care.")
    return {
        "region": REGION,
        "origin": origin,
        "nearest": _public(origin, nearest_h, nearest_d),
        "best": _public(origin, best_h, best_d),
        "same_hospital": same,
        "summary": summary,
        "all_hospitals": ranking,
        "method": ("Nearest = smallest straight-line distance. Best = highest evidence score (MG named on the "
                   "hospital's own pages 2-3, EMG/nerve-conduction lab 1-2, neurology department 1; max 6); "
                   "ties are broken by distance."),
        "disclaimer": DISCLAIMER,
    }
