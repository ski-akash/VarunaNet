import { useEffect, useRef, useState, type RefObject } from 'react'
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
  countryBorders: true,
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

// Same box, but padded with extra room north of India. Used only for the
// pitched globe fitBounds calls below. First attempt padded the southern
// (near-camera) edge instead, on the assumption that would push India up
// and let curvature "bulge in" below it -- backwards in practice: under
// a pitched perspective camera, near content eats far more screen space
// per degree of latitude than far content does, so padding the near
// side just forced a much bigger zoom-out and pushed India away, leaving
// a large empty gap beneath it (confirmed from a screenshot). Padding
// the *far* (north) edge instead is cheap in comparison, and lets India
// itself sit in the near/foreground of the frame -- bigger, closer, with
// the sphere's curve now visibly receding away behind it.
const INDIA_FIT_BOUNDS: [[number, number], [number, number]] = [
  [INDIA_BOUNDS[0][0], INDIA_BOUNDS[0][1]],
  [INDIA_BOUNDS[1][0], INDIA_BOUNDS[1][1] + 20],
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

// World country outlines -- Natural Earth's 1:50m admin-0 countries
// (public domain, the standard reference dataset for this, same quality
// tier as DataMeet for India). Simplified with mapshaper the same way as
// the India layers (`keep-shapes`, so small island nations don't get
// silently dropped). India itself is deliberately excluded from this
// file: it already has its own dedicated state/district layers above,
// and a second India polygon here would double up both the visuals and
// the click handling on the exact same shape. District-level detail
// isn't included for the rest of the world -- there's no single
// well-maintained open dataset at that granularity globally the way
// DataMeet provides for India specifically (project scope is Assam/the
// Brahmaputra basin, so that tradeoff is fine for now).
const WORLD_COUNTRIES_GEOJSON_URL = '/geo/world_countries.geojson'

// The "background" layer fills every pixel not covered by a country/
// state polygon -- on a globe, that's the ocean, so it reads oddly if
// it's not some shade of blue (the original off-white made every sea
// and lake look like blank, undefined space rather than water). Paired
// with a warm, muted land tone rather than reusing that same blue for
// countries -- the classic land/water split from physical atlases (and
// Natural Earth's own reference styling, the same project the world
// country and India district data already comes from), and it keeps
// land visually distinct from the severity legend's blue "safe" end
// (severityColor.ts) so the two don't get read as the same signal.
const OCEAN_COLOR = '#1c4966'
const LAND_FILL_COLOR = '#e3d5b8'
const LAND_HOVER_COLOR = '#c7b384'
// Boundary lines sit on top of land, not ocean -- needs its own tone
// (a muted brown, not white) to actually show up against the tan fill.
const BORDER_LINE_COLOR = '#9c8a66'

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
  // Renders the same flat choropleth wrapped onto an interactive 3D
  // sphere instead of a flat mercator projection -- confirmed directly
  // against the installed MapLibre GL version's own type definitions
  // that 'globe' is a real, supported projection (not assumed from
  // general knowledge, since this is a newer MapLibre feature). Global
  // vector data (our state/district GeoJSON) projects onto the sphere
  // the same way regardless of projection mode; nothing about the data
  // or layers below needs to change for this.
  projection: { type: 'globe' },
  // Without this, the space around the sphere just renders as flat
  // background color -- same as mercator -- so nothing at the edges
  // reads as "round". A dark sky plus a lighter atmosphere halo at the
  // horizon gives the globe's silhouette something to contrast against.
  // atmosphere-blend is interpolated by zoom (MapLibre's own recommended
  // pattern for globe) so the halo fades out once a state/district is
  // zoomed into and the view is effectively flat/mercator-like again.
  sky: {
    'sky-color': '#0b1026',
    'horizon-color': '#bfe3ff',
    'sky-horizon-blend': 0.6,
    'atmosphere-blend': [
      'interpolate',
      ['linear'],
      ['zoom'],
      0,
      1,
      4,
      0.7,
      7,
      0,
    ],
  },
  sources: {},
  layers: [
    {
      id: 'background',
      type: 'background',
      paint: { 'background-color': OCEAN_COLOR },
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
// isn't real yet). Land tone comes from LAND_FILL_COLOR/LAND_HOVER_COLOR
// above (kept blue previously, which fought with the ocean once the
// world map made the water/land distinction visible).
const STATE_FILL_COLOR = LAND_FILL_COLOR
const STATE_HOVER_COLOR = LAND_HOVER_COLOR
const STATE_SELECTED_OUTLINE_COLOR = '#f4a53a'
// A distinct accent (violet, not the state's amber) for district
// selection, so "you're looking at this district" reads as a different
// drill-down level from "you're looking at this state", not a
// same-meaning highlight one level down.
const DISTRICT_HOVER_FILL_COLOR = '#8b5cf6'
const DISTRICT_SELECTED_OUTLINE_COLOR = '#6d28d9'
const FADE_IN_MS = 900
const ZOOM_ANIMATION_MS = 1200

// Tilting the camera (pitch) is what actually sells the curvature here --
// looking straight down at a sphere (pitch 0) reads as a flat disk no
// matter how far zoomed out; angling the view toward the horizon is what
// makes the globe's surface visibly curve away, the same trick globe
// visualizations (Stripe's, GitHub's) use. That lets India stay fairly
// close-up (maxZoom higher than a plain top-down framing could get away
// with) while still reading clearly as a sphere, not a flat map.
//
// No bearing/roll here -- those rotate the camera itself and just spin
// or tilt the whole framed picture (tried both; neither read as "facing
// the viewer"). The "faces the viewer" feel instead comes from fitting
// to INDIA_FIT_BOUNDS (padded south) below, which shifts the camera's
// look-at point -- not its rotation -- south of India.
const INDIA_FIT_OPTIONS = { padding: 60, maxZoom: 3.8, pitch: 55 }

// Ambient auto-rotation for the globe: one full turn every 4 minutes --
// slow enough to read as "idle background motion", not a spinning toy.
// Only active at the whole-India view (no state drilled into) and paused
// the instant the user drags/rotates/pitches the globe themselves, so it
// never fights a real interaction.
const SPIN_DEGREES_PER_SECOND = 360 / 240
const SPIN_STEP_MS = 1000

function spinGlobe(
  map: MapLibreMap,
  selectedRef: RefObject<string | number | null>,
  selectedCountryRef: RefObject<string | number | null>,
  userInteractingRef: RefObject<boolean>,
) {
  if (
    userInteractingRef.current ||
    selectedRef.current !== null ||
    selectedCountryRef.current !== null
  ) {
    return
  }
  const center = map.getCenter()
  center.lng -= SPIN_DEGREES_PER_SECOND
  map.easeTo({ center, duration: SPIN_STEP_MS, easing: (t) => t })
}

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

// Mirrors addIndiaLayers' state-fill/border/selected-outline structure
// below, but one level shallower -- countries have no district-equivalent
// drill-down here, just hover, click-to-zoom, and a selected outline.
function addWorldLayers(
  map: MapLibreMap,
  selectedCountryRef: { current: string | number | null },
  onCountryClick: (feature: MapGeoJSONFeature) => void,
): void {
  map.addSource('world-countries', {
    type: 'geojson',
    data: WORLD_COUNTRIES_GEOJSON_URL,
    promoteId: 'name',
  })

  map.addLayer({
    id: 'world-country-fill',
    type: 'fill',
    source: 'world-countries',
    paint: {
      'fill-color': [
        'case',
        ['boolean', ['feature-state', 'hover'], false],
        STATE_HOVER_COLOR,
        STATE_FILL_COLOR,
      ],
      'fill-opacity': 0, // animated up to full below, same as the state layer
    },
  })

  map.addLayer({
    id: 'world-country-borders',
    type: 'line',
    source: 'world-countries',
    paint: {
      'line-color': BORDER_LINE_COLOR,
      'line-width': 1,
      'line-opacity': 0.5,
    },
  })

  map.addLayer({
    id: 'world-country-selected-outline',
    type: 'line',
    source: 'world-countries',
    paint: {
      'line-color': STATE_SELECTED_OUTLINE_COLOR,
      'line-width': ['case', ['boolean', ['feature-state', 'selected'], false], 3, 0],
    },
  })

  const start = performance.now()
  const targetOpacity = 0.92
  function animateFadeIn(now: number) {
    const progress = Math.min((now - start) / FADE_IN_MS, 1)
    map.setPaintProperty('world-country-fill', 'fill-opacity', progress * targetOpacity)
    if (progress < 1) {
      requestAnimationFrame(animateFadeIn)
    }
  }
  requestAnimationFrame(animateFadeIn)

  let hoveredCountryName: string | number | null = null
  map.on('mousemove', 'world-country-fill', (event) => {
    if (!event.features || event.features.length === 0) {
      return
    }
    const featureId = event.features[0].id
    if (featureId === undefined || featureId === hoveredCountryName) {
      return
    }
    if (hoveredCountryName !== null) {
      map.setFeatureState({ source: 'world-countries', id: hoveredCountryName }, { hover: false })
    }
    hoveredCountryName = featureId
    map.setFeatureState({ source: 'world-countries', id: featureId }, { hover: true })
    map.getCanvas().style.cursor = 'pointer'
  })
  map.on('mouseleave', 'world-country-fill', () => {
    if (hoveredCountryName !== null) {
      map.setFeatureState({ source: 'world-countries', id: hoveredCountryName }, { hover: false })
      hoveredCountryName = null
    }
    map.getCanvas().style.cursor = ''
  })

  map.on('click', 'world-country-fill', (event) => {
    if (!event.features || event.features.length === 0) {
      return
    }
    const feature = event.features[0]
    if (feature.id === undefined) {
      return
    }

    if (selectedCountryRef.current !== null) {
      map.setFeatureState(
        { source: 'world-countries', id: selectedCountryRef.current },
        { selected: false },
      )
    }
    selectedCountryRef.current = feature.id
    map.setFeatureState({ source: 'world-countries', id: feature.id }, { selected: true })

    map.fitBounds(boundingBoxOf(feature.geometry), {
      padding: 48,
      duration: ZOOM_ANIMATION_MS,
      pitch: 0,
      roll: 0,
    })
    onCountryClick(feature)
  })
}

function addIndiaLayers(
  map: MapLibreMap,
  selectedRef: { current: string | number | null },
  selectedDistrictRef: { current: string | number | null },
  onStateClick: (feature: MapGeoJSONFeature) => void,
  onDistrictClick: (feature: MapGeoJSONFeature) => void,
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

  // promoteId here is 'id', not 'name' -- district names are NOT
  // guaranteed unique across India (confirmed directly against the real
  // data: 10 duplicate names, including plain "East"/"North"/"South"/
  // "West", likely from Sikkim or a similar state). Using 'name' as the
  // feature-state key would mean setFeatureState on one "East" district
  // silently also affects every other district sharing that name --
  // 'id' is a synthetic per-feature index added specifically to avoid
  // this, since the source shapefile's own numeric census code wasn't
  // carried into the already-simplified/trimmed file this project ships.
  map.addSource('india-districts', {
    type: 'geojson',
    data: DISTRICTS_GEOJSON_URL,
    promoteId: 'id',
  })

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
      'line-color': BORDER_LINE_COLOR,
      'line-width': 0.6,
      'line-opacity': 0.6,
    },
  })

  // Hit-test layer for district hover/click -- fully transparent except
  // when hovered. Clicking is gated on a state already being selected
  // (see the click handler below): without that guard, clicking anywhere
  // in the full-India view would fire both this layer's click handler and
  // the state-fill layer's at the same point simultaneously (MapLibre
  // dispatches per-layer click handlers independently, not by
  // topmost-layer-wins), racing two conflicting fitBounds calls.
  map.addLayer({
    id: 'india-district-fill',
    type: 'fill',
    source: 'india-districts',
    paint: {
      'fill-color': DISTRICT_HOVER_FILL_COLOR,
      'fill-opacity': ['case', ['boolean', ['feature-state', 'hover'], false], 0.3, 0],
    },
  })

  map.addLayer({
    id: 'india-district-selected-outline',
    type: 'line',
    source: 'india-districts',
    paint: {
      'line-color': DISTRICT_SELECTED_OUTLINE_COLOR,
      'line-width': ['case', ['boolean', ['feature-state', 'selected'], false], 2.5, 0],
    },
  })

  map.addLayer({
    id: 'india-state-borders',
    type: 'line',
    source: 'india-states',
    paint: {
      'line-color': BORDER_LINE_COLOR,
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

    // A newly clicked state clears any district selection left over from
    // a previously drilled-into state -- otherwise a stale district
    // highlight (and stale "back to <district's state>" label) would
    // persist across an unrelated state selection.
    if (selectedDistrictRef.current !== null) {
      map.setFeatureState(
        { source: 'india-districts', id: selectedDistrictRef.current },
        { selected: false },
      )
      selectedDistrictRef.current = null
    }

    // pitch: 0 / roll: 0 resets the tilted, rotated "hero globe" idle
    // view back to a flat, straight-down framing once drilling into a
    // state -- that framing is meant to sell the globe at the whole-
    // India view, not persist into a close-up state read where it would
    // just look off-kilter.
    map.fitBounds(boundingBoxOf(feature.geometry), {
      padding: 48,
      duration: ZOOM_ANIMATION_MS,
      pitch: 0,
      roll: 0,
    })
    onStateClick(feature)
  })

  let hoveredDistrictId: string | number | null = null
  map.on('mousemove', 'india-district-fill', (event) => {
    if (!event.features || event.features.length === 0) {
      return
    }
    const featureId = event.features[0].id
    if (featureId === undefined || featureId === hoveredDistrictId) {
      return
    }
    if (hoveredDistrictId !== null) {
      map.setFeatureState({ source: 'india-districts', id: hoveredDistrictId }, { hover: false })
    }
    hoveredDistrictId = featureId
    map.setFeatureState({ source: 'india-districts', id: featureId }, { hover: true })
    // Only shows the pointer cursor once a state is selected -- matches
    // the click guard below, so the cursor doesn't promise an
    // interaction that clicking wouldn't actually perform yet.
    if (selectedRef.current !== null) {
      map.getCanvas().style.cursor = 'pointer'
    }
  })
  map.on('mouseleave', 'india-district-fill', () => {
    if (hoveredDistrictId !== null) {
      map.setFeatureState({ source: 'india-districts', id: hoveredDistrictId }, { hover: false })
      hoveredDistrictId = null
    }
    map.getCanvas().style.cursor = ''
  })

  map.on('click', 'india-district-fill', (event) => {
    // Drill-down only works one level at a time: India -> state -> its
    // districts. A state must already be selected.
    if (selectedRef.current === null) {
      return
    }
    if (!event.features || event.features.length === 0) {
      return
    }
    const feature = event.features[0]
    if (feature.id === undefined) {
      return
    }

    if (selectedDistrictRef.current !== null) {
      map.setFeatureState(
        { source: 'india-districts', id: selectedDistrictRef.current },
        { selected: false },
      )
    }
    selectedDistrictRef.current = feature.id
    map.setFeatureState({ source: 'india-districts', id: feature.id }, { selected: true })

    // Deliberately no fitBounds/zoom here, unlike the state click handler
    // above -- selecting a district highlights it (violet outline) and
    // updates the location label, but stays at whatever zoom level the
    // user was already looking at the state with, rather than jumping
    // the camera in further.
    onDistrictClick(feature)
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
  const selectedDistrictRef = useRef<string | number | null>(null)
  // A world country selection is a separate, parallel track from the
  // India state/district one above -- at most one of the two is ever
  // active, since clicking into either clears the other (see the click
  // handlers below).
  const selectedCountryRef = useRef<string | number | null>(null)
  // Retained so "back" from a selected district can return to exactly
  // the selected state's own framing, instead of re-deriving it or
  // falling back to the whole-India view and losing one drill-down level.
  const selectedStateBoundsRef = useRef<[[number, number], [number, number]] | null>(null)
  const userInteractingRef = useRef(false)
  const [selectedStateName, setSelectedStateName] = useState<string | null>(null)
  const [selectedDistrictName, setSelectedDistrictName] = useState<string | null>(null)
  const [selectedCountryName, setSelectedCountryName] = useState<string | null>(null)
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

    // Pause auto-rotation for the duration of any real user gesture, and
    // let it resume via the 'moveend' chain below once the gesture ends.
    // 'dragend'/'pitchend'/'rotateend' fire before 'moveend' for the same
    // gesture, so userInteractingRef is already false again by the time
    // spinGlobe re-checks it.
    map.on('dragstart', () => {
      userInteractingRef.current = true
    })
    map.on('dragend', () => {
      userInteractingRef.current = false
    })
    map.on('pitchstart', () => {
      userInteractingRef.current = true
    })
    map.on('pitchend', () => {
      userInteractingRef.current = false
    })
    map.on('rotatestart', () => {
      userInteractingRef.current = true
    })
    map.on('rotateend', () => {
      userInteractingRef.current = false
    })
    // Each spin step is itself a move, so its own 'moveend' re-triggers
    // this and keeps the rotation going indefinitely -- until the India-
    // level/no-interaction guard inside spinGlobe stops it.
    map.on('moveend', () => {
      spinGlobe(map, selectedRef, selectedCountryRef, userInteractingRef)
    })

    map.on('load', () => {
      // Added first so India's own layers (added right after) paint on
      // top -- doesn't actually matter for correctness since India is
      // excluded from the world dataset and the two never overlap
      // geographically, but keeps a sensible, predictable layer order.
      addWorldLayers(map, selectedCountryRef, (feature) => {
        // A country selection and an India state/district selection are
        // mutually exclusive (see selectedCountryRef's own comment) --
        // clicking a country while a state/district was selected clears
        // that stale selection the same way a new state click clears a
        // stale district one below.
        if (selectedDistrictRef.current !== null) {
          map.setFeatureState(
            { source: 'india-districts', id: selectedDistrictRef.current },
            { selected: false },
          )
          selectedDistrictRef.current = null
          setSelectedDistrictName(null)
        }
        if (selectedRef.current !== null) {
          map.setFeatureState({ source: 'india-states', id: selectedRef.current }, { selected: false })
          selectedRef.current = null
          setSelectedStateName(null)
        }
        setSelectedCountryName(String(feature.properties?.name ?? ''))
      })

      addIndiaLayers(
        map,
        selectedRef,
        selectedDistrictRef,
        (feature) => {
          // Same mutual-exclusion clearing as above, in the other direction.
          if (selectedCountryRef.current !== null) {
            map.setFeatureState(
              { source: 'world-countries', id: selectedCountryRef.current },
              { selected: false },
            )
            selectedCountryRef.current = null
            setSelectedCountryName(null)
          }
          selectedStateBoundsRef.current = boundingBoxOf(feature.geometry)
          setSelectedStateName(String(feature.properties?.name ?? ''))
          setSelectedDistrictName(null)
        },
        (feature) => {
          setSelectedDistrictName(String(feature.properties?.name ?? ''))
        },
      )
      map.fitBounds(INDIA_FIT_BOUNDS, { ...INDIA_FIT_OPTIONS, duration: 2200 })
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
    // and addWorldLayers/addIndiaLayers have run, so a toggle flipped in
    // that (normally very short) window would otherwise hit a "layer not
    // found" error.
    if (map.getLayer('world-country-borders')) {
      map.setLayoutProperty(
        'world-country-borders',
        'visibility',
        next.countryBorders ? 'visible' : 'none',
      )
    }
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

  // One level of "back" at a time: a selected district returns to its
  // state's own framing; a selected state (no district) returns to the
  // world view; a selected world country (no India state/district
  // selected -- the two tracks are mutually exclusive) also returns to
  // the world view.
  function handleBack() {
    const map = mapRef.current
    if (map === null) {
      return
    }

    if (selectedDistrictRef.current !== null) {
      map.setFeatureState(
        { source: 'india-districts', id: selectedDistrictRef.current },
        { selected: false },
      )
      selectedDistrictRef.current = null
      setSelectedDistrictName(null)
      if (selectedStateBoundsRef.current !== null) {
        map.fitBounds(selectedStateBoundsRef.current, { padding: 48, duration: ZOOM_ANIMATION_MS })
      }
      return
    }

    if (selectedCountryRef.current !== null) {
      map.setFeatureState(
        { source: 'world-countries', id: selectedCountryRef.current },
        { selected: false },
      )
      selectedCountryRef.current = null
      setSelectedCountryName(null)
      map.fitBounds(INDIA_FIT_BOUNDS, { ...INDIA_FIT_OPTIONS, duration: ZOOM_ANIMATION_MS })
      return
    }

    if (selectedRef.current !== null) {
      map.setFeatureState({ source: 'india-states', id: selectedRef.current }, { selected: false })
      selectedRef.current = null
    }
    setSelectedStateName(null)
    map.fitBounds(INDIA_FIT_BOUNDS, { ...INDIA_FIT_OPTIONS, duration: ZOOM_ANIMATION_MS })
  }

  const backButtonLabel =
    selectedDistrictName !== null
      ? selectedStateName
      : selectedStateName !== null || selectedCountryName !== null
        ? 'World'
        : null

  // "Where am I" -- separate from the back button, which names the
  // *target* of going back, not the current selection. Without this, the
  // district's name only ever showed up as an unlabeled violet outline
  // on the map itself.
  const currentLocationLabel =
    selectedDistrictName !== null
      ? `${selectedDistrictName}, ${selectedStateName}`
      : (selectedStateName ?? selectedCountryName)

  return (
    <div className="map-view-wrapper">
      <div ref={containerRef} className="map-view" data-testid="map-view" />
      {backButtonLabel !== null && (
        <div className="map-nav">
          <button type="button" className="back-to-india" onClick={handleBack}>
            ← {backButtonLabel}
          </button>
          {currentLocationLabel !== null && (
            <span className="current-location">{currentLocationLabel}</span>
          )}
        </div>
      )}
      <LayerToggles value={layerToggles} onChange={handleLayerTogglesChange} />
      <SeverityLegend />
      <FloodReportBadge />
    </div>
  )
}
