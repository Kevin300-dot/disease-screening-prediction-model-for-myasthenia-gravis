// South Chennai localities for the "Your area" picker. Keys must match AREAS in
// backend/mg_predictor/mg/hospitals.py exactly (tests/test_pipeline.py checks that they agree).
// MyaScreen's hospital finder covers South Chennai only.

export const REGION = "South Chennai";

export const southChennaiAreas = [
  { key: "adyar", label: "Adyar" },
  { key: "alandur", label: "Alandur" },
  { key: "besant_nagar", label: "Besant Nagar" },
  { key: "chromepet", label: "Chromepet" },
  { key: "guindy", label: "Guindy" },
  { key: "kotturpuram", label: "Kotturpuram" },
  { key: "kovilambakkam", label: "Kovilambakkam" },
  { key: "madipakkam", label: "Madipakkam" },
  { key: "medavakkam", label: "Medavakkam" },
  { key: "nanganallur", label: "Nanganallur" },
  { key: "pallavaram", label: "Pallavaram" },
  { key: "pallikaranai", label: "Pallikaranai" },
  { key: "perumbakkam", label: "Perumbakkam" },
  { key: "perungudi", label: "Perungudi" },
  { key: "saidapet", label: "Saidapet" },
  { key: "selaiyur", label: "Selaiyur" },
  { key: "sholinganallur", label: "Sholinganallur" },
  { key: "tambaram", label: "Tambaram" },
  { key: "taramani", label: "Taramani" },
  { key: "thiruvanmiyur", label: "Thiruvanmiyur" },
  { key: "thoraipakkam", label: "Thoraipakkam" },
  { key: "velachery", label: "Velachery" },
] as const;

// Same box as BOUNDS in the backend; used to reject GPS positions outside South Chennai.
export const SOUTH_CHENNAI_BOUNDS = { latMin: 12.86, latMax: 13.04, lngMin: 80.07, lngMax: 80.3 };

export function insideSouthChennai(lat: number, lng: number): boolean {
  const b = SOUTH_CHENNAI_BOUNDS;
  return lat >= b.latMin && lat <= b.latMax && lng >= b.lngMin && lng <= b.lngMax;
}
