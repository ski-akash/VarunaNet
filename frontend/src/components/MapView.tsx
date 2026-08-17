import { useEffect, useRef, useState } from 'react'
import {
  Map as MapLibreMap,
  NavigationControl,
  type StyleSpecification,
  type MapGeoJSONFeature,
} from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
import { CURRENT_FLOOD_REPORT } from '../lib/currentFloodReports'
import FloodReportBadge from './FloodReportBadge'
import LayerToggles, { type LayerToggleState } from './LayerToggles'
import SeverityLegend from './SeverityLegend'

const DEFAULT_LAYER_TOGGLES: LayerToggleState = {
  stateBorders: true,
  districtBorders: true,
}

// India's real bounding box (southwest, northeast corners), computed
// directly from the actual state polygon data below -- not eyeballed --
// so the view genuinely frames the whole country (Ladakh at the north
// tip, Kanyakumari at the south, the northeast states, all of it),
// rather than a guessed center/zoom that happened to clip the edges.
const INDIA_BOUNDS: [[number, number], [number, number]] = [
  [68.12, 6.78],
  [97.39, 37.08],
]

// State/UT boundary polygons -- from DataMeet (github.com/datameet/maps),
// an Indian community-maintained open data source. All 36 current
// states/UTs, including Telangana (2014) and Ladakh (2019) -- an earlier
// GADM-derived dataset was missing both. Converted from the source
// Shapefile and simplified with mapshaper (~41MB -> ~66KB; `keep-shapes`
// was needed to stop small territories like Lakshadweep from being
// silently dropped). Served from public/ as a static asset, not bundled.
const STATES_GEOJSON_URL = '/geo/india_states.geojson'

// District boundary lines, also from DataMeet -- their 2011 Census
// district dataset (India's most recent census with district mapping;
// 2021's was delayed). Boundary *shapes* don't change just because a
// district's state assignment did, so these lines are safe to show now.
// What's deliberately NOT carried over from the source data: each
// district's state attribution. The 2011 data predates the Telangana
// (2014) and Ladakh (2019) splits, so districts are still grouped under
// their pre-split states (e.g. Adilabad still shows as Andhra Pradesh,
// not Telangana) -- the same currency problem the state dataset had,
// just not corrected yet at this level. Fine for drawing boundary lines
// only; would need fixing first if a future feature (click-a-district,
// group-by-state) actually depends on that attribute being right.
const DISTRICTS_GEOJSON_URL = '/geo/india_districts.geojson'

// No raster basemap underneath (no OpenStreetMap tiles, no satellite
// imagery, no roads/terrain) -- this is a flat, illustrated choropleth,
// not a navigation-style map. A real basemap was tried first and was the
// wrong call for this project: spec section 7 asks for an India state/
// district choropleth colored by flood severity, which is inherently a
// flat design (like a data dashboard, not Google Maps), and a photoreal
// basemap underneath just competes with the severity colors that are the
// actual point once real model output exists. The only "layer" here is
// a solid background color; the state shapes themselves are the map.
const FLAT_STYLE: StyleSpecification = {
  version: 8,
  sources: {},
  layers: [
    {
      id: 'background',
      type: 'background',
      paint: { 'background-color': '#eef3f8' },
    },
  ],
}

// Placeholder: every state gets the same flat color for now. Spec
// section 7 wants states colored by flood severity on a sequential
// scale -- that needs real model output (Phase 5), which doesn't exist
// yet. Coloring states with fake severity here would be exactly the kind
// of invented-looking number this project explicitly refuses to show
// anywhere (spec section 6.1's grounding rule is about the AI layer, but
// the same principle applies to the map: don't visually imply data that
// isn't real yet).
const STATE_FILL_COLOR = '#2f6690'
const STATE_HOVER_COLOR = '#1c4a6e'
const STATE_SELECTED_OUTLINE_COLOR = '#f4a53a'
const FADE_IN_MS = 900
const ZOOM_ANIMATION_MS = 1200

// A GeoJSON Polygon's coordinates are number[][][] (rings of points),
// a MultiPolygon's are number[][][][] (polygons of rings of points) --
// arbitrarily nested arrays that bottom out in a [lon, lat] pair.
type NestedPosition = number[] | NestedPosition[]

// Flattens a Polygon/MultiPolygon's coordinates into a [[minLon, minLat],
// [maxLon, maxLat]] bounding box. Written locally instead of pulling in
// @turf/bbox as a runtime dependency for one small computation -- the
// heavier geometry work (union, masking) already happened once, offline,
// when the source data was prepared; this is the one piece of geometry
// math this project actually needs at runtime.
function boundingBoxOf(geometry: MapGeoJSONFeature['geometry']): [[number, number], [number, number]] {
  let minLon = Infinity
  let minLat = Infinity
  let maxLon = -Infinity
  let maxLat = -Infinity

  function visit(coords: NestedPosition): void {
    if (typeof coords[0] === 'number') {
      const [lon, lat] = coords as number[]
      minLon = Math.min(minLon, lon)
      maxLon = Math.max(maxLon, lon)
      minLat = Math.min(minLat, lat)
      maxLat = Math.max(maxLat, lat)
      return
    }
    for (const item of coords as NestedPosition[]) {
      visit(item)
    }
  }
  visit('coordinates' in geometry ? geometry.coordinates : [])

  return [
    [minLon, minLat],
    [maxLon, maxLat],
  ]
}

function addIndiaLayers(
  map: MapLibreMap,
  selectedRef: { current: string | number | null },
  onStateClick: (feature: MapGeoJSONFeature) => void,
): void {
  // promoteId: each state's own name becomes its feature id, which is
  // what setFeatureState below needs to target one specific state rather
  // than the whole layer.
  map.addSource('india-states', {
    type: 'geojson',
    data: STATES_GEOJSON_URL,
    promoteId: 'name',
  })

  map.addLayer({
    id: 'india-state-fill',
    type: 'fill',
    source: 'india-states',
    paint: {
      'fill-color': [
        'case',
        ['boolean', ['feature-state', 'hover'], false],
        STATE_HOVER_COLOR,
        STATE_FILL_COLOR,
      ],
      'fill-opacity': 0, // animated up to full below -- the map's entrance animation
    },
  })

  map.addSource('india-districts', { type: 'geojson', data: DISTRICTS_GEOJSON_URL })

  // Real, sourced, current news reports (see lib/currentFloodReports.ts)
  // -- deliberately NOT the severity-gradient palette (severityColor.ts):
  // that scale is reserved for real SAR-measured extent, which this
  // isn't. Cross-hatch-style dashed outline plus a light fill, so it
  // reads as "flagged from a report" rather than "precisely measured",
  // and stays visually distinct from every other layer on this map.
  map.addLayer({
    id: 'reported-flood-affected-fill',
    type: 'fill',
    source: 'india-districts',
    filter: ['in', ['get', 'name'], ['literal', [...CURRENT_FLOOD_REPORT.affectedDistricts]]],
    paint: {
      'fill-color': '#e31a1c',
      'fill-opacity': 0.35,
    },
  })
  map.addLayer({
    id: 'reported-flood-affected-outline',
    type: 'line',
    source: 'india-districts',
    filter: ['in', ['get', 'name'], ['literal', [...CURRENT_FLOOD_REPORT.affectedDistricts]]],
    paint: {
      'line-color': '#7f0000',
      'line-width': 1.5,
      'line-dasharray': [1, 1],
    },
  })

  // Added before the state-border layer below (MapLibre draws later
  // layers on top), so where a district edge coincides with a state
  // edge, the more prominent dashed state line wins visually -- district
  // lines only really stand out on internal, within-state boundaries,
  // keeping the state/district hierarchy visually readable rather than
  // both styles fighting for the same pixels.
  map.addLayer({
    id: 'india-district-borders',
    type: 'line',
    source: 'india-districts',
    paint: {
      'line-color': '#eef3f8',
      'line-width': 0.6,
      'line-opacity': 0.6,
    },
  })

  map.addLayer({
    id: 'india-state-borders',
    type: 'line',
    source: 'india-states',
    paint: {
      'line-color': '#eef3f8',
      'line-width': 1.5,
      'line-dasharray': [2, 1.5],
    },
  })

  // A distinct highlighted outline for whichever state is currently
  // drilled into -- separate from the hover layer above, so "this is the
  // state I'm looking at" reads differently from "this is what my cursor
  // happens to be over".
  map.addLayer({
    id: 'india-state-selected-outline',
    type: 'line',
    source: 'india-states',
    paint: {
      'line-color': STATE_SELECTED_OUTLINE_COLOR,
      'line-width': ['case', ['boolean', ['feature-state', 'selected'], false], 3, 0],
    },
  })

  // Fade the states in from transparent to fully opaque -- an animated
  // build-up rather than the shapes just appearing instantly.
  const start = performance.now()
  const targetOpacity = 0.92
  function animateFadeIn(now: number) {
    const progress = Math.min((now - start) / FADE_IN_MS, 1)
    map.setPaintProperty('india-state-fill', 'fill-opacity', progress * targetOpacity)
    if (progress < 1) {
      requestAnimationFrame(animateFadeIn)
    }
  }
  requestAnimationFrame(animateFadeIn)

  let hoveredStateName: string | number | null = null
  map.on('mousemove', 'india-state-fill', (event) => {
    if (!event.features || event.features.length === 0) {
      return
    }
    const featureId = event.features[0].id
    if (featureId === undefined || featureId === hoveredStateName) {
      return
    }
    if (hoveredStateName !== null) {
      map.setFeatureState({ source: 'india-states', id: hoveredStateName }, { hover: false })
    }
    hoveredStateName = featureId
    map.setFeatureState({ source: 'india-states', id: featureId }, { hover: true })
    map.getCanvas().style.cursor = 'pointer'
  })
  map.on('mouseleave', 'india-state-fill', () => {
    if (hoveredStateName !== null) {
      map.setFeatureState({ source: 'india-states', id: hoveredStateName }, { hover: false })
      hoveredStateName = null
    }
    map.getCanvas().style.cursor = ''
  })

  map.on('click', 'india-state-fill', (event) => {
    if (!event.features || event.features.length === 0) {
      return
    }
    const feature = event.features[0]
    if (feature.id === undefined) {
      return
    }

    if (selectedRef.current !== null) {
      map.setFeatureState({ source: 'india-states', id: selectedRef.current }, { selected: false })
    }
    selectedRef.current = feature.id
    map.setFeatureState({ source: 'india-states', id: feature.id }, { selected: true })

    map.fitBounds(boundingBoxOf(feature.geometry), { padding: 48, duration: ZOOM_ANIMATION_MS })
    onStateClick(feature)
  })
}

interface MapViewProps {
  // Notified whenever the selected state changes (click, or back-to-India
  // clearing it) -- lets App.tsx offer a "generate report for this state"
  // action without MapView needing to know anything about reports itself.
  onSelectionChange?: (stateName: string | null) => void
}

export default function MapView({ onSelectionChange }: MapViewProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const mapRef = useRef<MapLibreMap | null>(null)
  const selectedRef = useRef<string | number | null>(null)
  const [selectedStateName, setSelectedStateName] = useState<string | null>(null)
  const [layerToggles, setLayerToggles] = useState<LayerToggleState>(DEFAULT_LAYER_TOGGLES)

  useEffect(() => {
    onSelectionChange?.(selectedStateName)
  }, [selectedStateName, onSelectionChange])

  useEffect(() => {
    if (containerRef.current === null || mapRef.current !== null) {
      return
    }

    // Starts zoomed out, then flies in to India's exact bounds once
    // loaded -- an intro animation rather than snapping straight to the
    // final framing.
    const map = new MapLibreMap({
      container: containerRef.current,
      style: FLAT_STYLE,
      center: [82.0, 22.0],
      zoom: 2,
    })
    map.addControl(new NavigationControl(), 'top-right')
    map.on('load', () => {
      addIndiaLayers(map, selectedRef, (feature) => {
        setSelectedStateName(String(feature.properties?.name ?? ''))
      })
      map.fitBounds(INDIA_BOUNDS, { padding: 24, duration: 2200 })
    })
    mapRef.current = map

    // Deliberately no cleanup that calls map.remove() here. React's
    // StrictMode (see main.tsx) double-invokes effects in development --
    // mount, cleanup, mount again -- specifically to catch bugs like the
    // one this caused: calling .remove() while MapLibre is still
    // mid-initialization (loading its style, setting up WebGL) leaves the
    // canvas in a broken state for the second map instance created right
    // after. Confirmed directly from a screenshot: nav controls rendered
    // fine (plain DOM overlays, unaffected) but the map canvas itself
    // stayed permanently blank. MapView is a top-level, effectively
    // singleton component for this dashboard's lifetime -- it's never
    // conditionally unmounted -- so skipping teardown here is a
    // deliberate, documented tradeoff, not an oversight.
  }, [])

  function handleLayerTogglesChange(next: LayerToggleState) {
    setLayerToggles(next)
    const map = mapRef.current
    if (map === null) {
      return
    }
    // Guarded with getLayer: the toggle panel renders immediately, but
    // the actual layers only exist once the map's 'load' event has fired
    // and addIndiaLayers has run, so a toggle flipped in that (normally
    // very short) window would otherwise hit a "layer not found" error.
    if (map.getLayer('india-state-borders')) {
      map.setLayoutProperty(
        'india-state-borders',
        'visibility',
        next.stateBorders ? 'visible' : 'none',
      )
    }
    if (map.getLayer('india-district-borders')) {
      map.setLayoutProperty(
        'india-district-borders',
        'visibility',
        next.districtBorders ? 'visible' : 'none',
      )
    }
  }

  function handleBackToIndia() {
    const map = mapRef.current
    if (map === null) {
      return
    }
    if (selectedRef.current !== null) {
      map.setFeatureState({ source: 'india-states', id: selectedRef.current }, { selected: false })
      selectedRef.current = null
    }
    setSelectedStateName(null)
    map.fitBounds(INDIA_BOUNDS, { padding: 24, duration: ZOOM_ANIMATION_MS })
  }

  return (
    <div className="map-view-wrapper">
      <div ref={containerRef} className="map-view" data-testid="map-view" />
      {selectedStateName !== null && (
        <button type="button" className="back-to-india" onClick={handleBackToIndia}>
          ← {selectedStateName}
        </button>
      )}
      <LayerToggles value={layerToggles} onChange={handleLayerTogglesChange} />
      <SeverityLegend />
      <FloodReportBadge />
    </div>
  )
}
