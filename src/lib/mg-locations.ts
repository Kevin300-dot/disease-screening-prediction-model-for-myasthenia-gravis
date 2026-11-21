// Chennai localities for the "Your area" picker. Keys must match AREAS in
// backend/mg_predictor/mg/hospitals.py exactly (tests/test_pipeline.py checks that they agree).
// MyaScreen's hospital finder covers Chennai.

export const REGION = "Chennai";

export const chennaiAreas = [
  { key: "adyar", label: "Adyar" },
  { key: "alandur", label: "Alandur" },
  { key: "alwarpet", label: "Alwarpet" },
  { key: "ambattur", label: "Ambattur" },
  { key: "anna_nagar", label: "Anna Nagar" },
  { key: "ashok_nagar", label: "Ashok Nagar" },
  { key: "avadi", label: "Avadi" },
  { key: "besant_nagar", label: "Besant Nagar" },
  { key: "chromepet", label: "Chromepet" },
  { key: "egmore", label: "Egmore" },
  { key: "ekkaduthangal", label: "Ekkaduthangal" },
  { key: "george_town", label: "George Town" },
  { key: "guindy", label: "Guindy" },
  { key: "kk_nagar", label: "K.K. Nagar" },
  { key: "kilpauk", label: "Kilpauk" },
  { key: "kodambakkam", label: "Kodambakkam" },
  { key: "kolathur", label: "Kolathur" },
  { key: "kotturpuram", label: "Kotturpuram" },
  { key: "kovilambakkam", label: "Kovilambakkam" },
  { key: "madipakkam", label: "Madipakkam" },
  { key: "manapakkam", label: "Manapakkam" },
  { key: "medavakkam", label: "Medavakkam" },
  { key: "mogappair", label: "Mogappair" },
  { key: "mylapore", label: "Mylapore" },
  { key: "nanganallur", label: "Nanganallur" },
  { key: "nungambakkam", label: "Nungambakkam" },
  { key: "pallavaram", label: "Pallavaram" },
  { key: "pallikaranai", label: "Pallikaranai" },
  { key: "park_town", label: "Park Town" },
  { key: "perambur", label: "Perambur" },
  { key: "perumbakkam", label: "Perumbakkam" },
  { key: "perungudi", label: "Perungudi" },
  { key: "porur", label: "Porur" },
  { key: "royapettah", label: "Royapettah" },
  { key: "royapuram", label: "Royapuram" },
  { key: "saidapet", label: "Saidapet" },
  { key: "selaiyur", label: "Selaiyur" },
  { key: "sholinganallur", label: "Sholinganallur" },
  { key: "t_nagar", label: "T. Nagar" },
  { key: "tambaram", label: "Tambaram" },
  { key: "taramani", label: "Taramani" },
  { key: "thiruvanmiyur", label: "Thiruvanmiyur" },
  { key: "thiruvottiyur", label: "Thiruvottiyur" },
  { key: "thoraipakkam", label: "Thoraipakkam" },
  { key: "tondiarpet", label: "Tondiarpet" },
  { key: "triplicane", label: "Triplicane" },
  { key: "vadapalani", label: "Vadapalani" },
  { key: "valasaravakkam", label: "Valasaravakkam" },
  { key: "velachery", label: "Velachery" },
  { key: "villivakkam", label: "Villivakkam" },
] as const;

// Same box as BOUNDS in the backend; used to reject GPS positions outside Chennai.
export const CHENNAI_BOUNDS = { latMin: 12.8, latMax: 13.25, lngMin: 80.05, lngMax: 80.35 };

export function insideChennai(lat: number, lng: number): boolean {
  const b = CHENNAI_BOUNDS;
  return lat >= b.latMin && lat <= b.latMax && lng >= b.lngMin && lng <= b.lngMax;
}
