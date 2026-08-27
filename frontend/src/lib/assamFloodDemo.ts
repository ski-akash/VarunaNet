// The real district-level result data/build_assam_demo.py produces: a
// genuine Sentinel-1 pair (Google Earth Engine, COPERNICUS/S1_GRD sigma0)
// over a real Assam AOI (the Majuli/Lakhimpur/Jorhat area) during the
// actual 2020 monsoon flood, scored with this project's own validated
// classical baseline (Otsu + HAND/slope + JRC permanent-water removal --
// NOT the untrained local demo CNN checkpoint, which would report zero
// flood on real data). Fetched as a static asset the same way
// assam_districts.geojson already is.
//
// Honest scope, stated here once rather than re-litigated at every call
// site: this is one small AOI (~16km box), not full-state coverage, so
// only the districts that AOI actually intersects (Lakhimpur, Jorhat as
// of the real run) carry a number -- every other district has none, and
// showing them as "0% flooded" would overclaim full-state monitoring this
// project doesn't have yet.
export interface AssamFloodDistrict {
  name: string
  flooded_hectares: number
  flooded_percent: number
}

export interface AssamFloodDemo {
  scene_id: string
  dry_reference_scene_id: string
  source: string
  method: string
  aoi_bounds: [number, number, number, number]
  processed_at: string
  water_pixel_fraction: number
  flood_pixel_fraction: number
  flood_polygons: number
  districts_total: number
  districts_affected: number
  total_flooded_hectares: number
  worst_affected: AssamFloodDistrict[]
}

const ASSAM_FLOOD_DEMO_URL = '/data/assam_flood_demo.json'

export async function fetchAssamFloodDemo(signal?: AbortSignal): Promise<AssamFloodDemo | null> {
  const res = await fetch(ASSAM_FLOOD_DEMO_URL, { signal })
  if (!res.ok) return null
  return (await res.json()) as AssamFloodDemo
}
