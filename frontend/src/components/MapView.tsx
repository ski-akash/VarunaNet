import { useEffect, useRef, useState } from 'react'
import {
  Map as MapLibreMap,
  NavigationControl,
  type ExpressionSpecification,
  type StyleSpecification,
} from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
import AssamFloodDemoBadge from './AssamFloodDemoBadge'
import LayerToggles from './LayerToggles'
import SeverityLegend from './SeverityLegend'
import { fetchAssamFloodDemo, type AssamFloodDemo } from '../lib/assamFloodDemo'
import { severityColor } from '../lib/severityColor'

// Assam's real bounding box (southwest, northeast corners), read straight
// off the state polygon in assam_state.geojson rather than eyeballed, so
// the view frames the whole state -- Dhubri at the western end of the
// Brahmaputra valley through to Tinsukia in the east.
const ASSAM_BOUNDS: [[number, number], [number, number]] = [
  [89.72, 24.136],
  [96.018, 27.963],
]
const ASSAM_FIT_OPTIONS = { padding: 32, maxZoom: 9 }

// Assam only -- the state outline and its 27 districts, and nothing else.
// No neighbouring states, no other countries, no ocean, no basemap.
//
// Both files come from DataMeet's full-resolution 2011 Census district
// shapefile (Districts/Census_2011/2011_Dist.shp), selected on its own
// ST_NM field and re-simplified with mapshaper for *state* zoom rather
// than the country zoom the previous all-India extracts were built for:
// the outline went from 128 vertices to ~2,000, which is the difference
// between a visibly faceted approximation and a real boundary.
//
// The state outline is the 27 districts dissolved together, not a
// separate state polygon, so the outline and the district edges are
// derived from exactly the same geometry and cannot disagree.
const ASSAM_STATE_GEOJSON_URL = '/geo/assam_state.geojson'
const ASSAM_DISTRICTS_GEOJSON_URL = '/geo/assam_districts.geojson'
// Real river centerlines from OpenStreetMap (waterway=river ways within
// Assam, fetched via the Overpass API -- © OpenStreetMap contributors,
// ODbL, same licence-requires-attribution category as DataMeet's
// boundaries below), filtered to the Brahmaputra and its 14 longest named
// tributaries by real total length (not hand-picked) and simplified for
// state-zoom display, the same way the district/state boundary files are
// pre-built rather than fetched live in the browser.
const ASSAM_RIVERS_GEOJSON_URL = '/geo/assam_rivers.geojson'
// The real vectorized flood polygons data/build_assam_statewide.py (or,
// for the single-AOI proof, data/build_assam_demo.py) writes -- per-pixel
// Otsu+HAND flood extent, not the district-level percentage the severity
// fill/badge use. Same file the district coloring's summary JSON sits
// next to, loaded as a second, independently toggleable layer.
const ASSAM_FLOOD_EXTENT_GEOJSON_URL = '/data/assam_flood_demo.geojson'

// Everything outside Assam is empty black space, by intent -- this is a
// single-subject map, not an atlas, so there is no land/water styling and
// no surrounding geography to colour. The state reads as a shape floating
// on the dashboard's own dark background.
const EMPTY_SPACE_COLOR = '#0b0b0d'
const DISTRICT_FILL_COLOR = '#232733'
const DISTRICT_HOVER_COLOR = '#33394a'
const DISTRICT_SELECTED_COLOR = '#4a5570'
const DISTRICT_BORDER_COLOR = '#5a6275'
const STATE_OUTLINE_COLOR = '#e6e8ee'
// Two distinct colors, not one flat blue for every river: the Brahmaputra
// (flagged `major` in the source data) is the subject of the whole
// project and reads brighter/thicker; its tributaries are a dimmer,
// thinner blue so they're visible without competing with it.
const RIVER_COLOR_MAJOR = '#4fc3f7'
const RIVER_COLOR_TRIBUTARY = '#2c6e8c'
// Distinct from both the district severity palette and the river blues --
// this is per-pixel flood extent, a different kind of claim (a measured
// boundary, not a district-level share), so it reads as its own layer
// rather than blending into either.
const FLOOD_EXTENT_FILL_COLOR = '#e2636b'

const BASE_STYLE: StyleSpecification = {
  version: 8,
  projection: { type: 'mercator' },
  sources: {},
  layers: [
    {
      id: 'background',
      type: 'background',
      paint: { 'background-color': EMPTY_SPACE_COLOR },
    },
  ],
}

interface MapViewProps {
  // Notified whenever the selected district changes (click, or the back
  // button clearing it), so App.tsx can offer a report for it without
  // MapView needing to know anything about reports itself.
  onSelectionChange?: (districtName: string | null) => void
}

export default function MapView({ onSelectionChange }: MapViewProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const mapRef = useRef<MapLibreMap | null>(null)
  const selectedDistrictRef = useRef<string | number | null>(null)
  const [selectedDistrictName, setSelectedDistrictName] = useState<string | null>(null)
  const [floodDemo, setFloodDemo] = useState<AssamFloodDemo | null>(null)
  const [layersChecked, setLayersChecked] = useState<Record<string, boolean>>({
    'flood-extent': false,
  })

  function toggleLayer(id: string) {
    setLayersChecked((prev) => {
      const next = { ...prev, [id]: !prev[id] }
      const map = mapRef.current
      if (map !== null && map.getLayer('flood-extent-fill')) {
        map.setLayoutProperty('flood-extent-fill', 'visibility', next[id] ? 'visible' : 'none')
      }
      return next
    })
  }

  useEffect(() => {
    onSelectionChange?.(selectedDistrictName)
  }, [selectedDistrictName, onSelectionChange])

  useEffect(() => {
    if (containerRef.current === null || mapRef.current !== null) {
      return
    }

    const map = new MapLibreMap({
      container: containerRef.current,
      style: BASE_STYLE,
      bounds: ASSAM_BOUNDS,
      fitBoundsOptions: ASSAM_FIT_OPTIONS,
      // The boundary data is DataMeet's, under CC BY 4.0, whose terms ask
      // that the dataset be named and linked wherever it's shown -- so
      // this credit is a licence condition, not a courtesy, and it has to
      // stay for as long as the map draws those shapes. (MapLibre's own
      // wordmark, which the control shows by default, is BSD-3-Clause:
      // that licence is satisfied by the notice carried in the bundled
      // source, so the on-map wordmark is convention rather than a
      // requirement.) compact:true keeps it as a single (i) disclosure
      // rather than a permanent bar of text across the map.
      attributionControl: {
        // false, not true: compact:true is the collapsed (i)-toggle form,
        // which needs a click to open and a click to close -- the source
        // list should just always be readable, no button.
        compact: false,
        customAttribution:
          'Boundaries: <a href="https://github.com/datameet/maps" target="_blank" rel="noreferrer">DataMeet</a>, Census 2011 (CC BY 4.0) · Rivers: <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noreferrer">OpenStreetMap</a> contributors (ODbL)',
      },
    })
    map.addControl(new NavigationControl(), 'top-right')

    // 'style.load', not 'load'. With this style (no remote tiles), the
    // map's 'load' event never fires at all -- confirmed by logging both
    // -- so anything registered on it silently never runs, which is what
    // left the map empty on a black background. 'style.load' is also the
    // event MapLibre documents for adding sources and layers.
    map.on('style.load', () => {
      // Districts carry the fill and all the interaction; the state layer
      // above them is just an outline, so the two never fight over a click.
      map.addSource('assam-districts', {
        type: 'geojson',
        data: ASSAM_DISTRICTS_GEOJSON_URL,
        // District names are not unique across India, so the extract gives
        // every feature its own numeric id and that is what gets promoted.
        promoteId: 'id',
      })
      map.addSource('assam-state', { type: 'geojson', data: ASSAM_STATE_GEOJSON_URL })
      map.addSource('assam-rivers', { type: 'geojson', data: ASSAM_RIVERS_GEOJSON_URL })
      map.addSource('assam-flood-extent', { type: 'geojson', data: ASSAM_FLOOD_EXTENT_GEOJSON_URL })

      map.addLayer({
        id: 'district-fill',
        type: 'fill',
        source: 'assam-districts',
        paint: {
          'fill-color': [
            'case',
            ['boolean', ['feature-state', 'selected'], false],
            DISTRICT_SELECTED_COLOR,
            ['boolean', ['feature-state', 'hover'], false],
            DISTRICT_HOVER_COLOR,
            DISTRICT_FILL_COLOR,
          ],
          // Set outright rather than animated up from 0. The fade-in that
          // used to be here drove opacity from a requestAnimationFrame
          // loop, which left the whole map invisible whenever that loop
          // didn't run or threw partway through.
          'fill-opacity': 1,
        },
      })

      map.addLayer({
        id: 'district-borders',
        type: 'line',
        source: 'assam-districts',
        paint: { 'line-color': DISTRICT_BORDER_COLOR, 'line-width': 0.8 },
      })

      map.addLayer({
        id: 'state-outline',
        type: 'line',
        source: 'assam-state',
        paint: { 'line-color': STATE_OUTLINE_COLOR, 'line-width': 1.6 },
      })

      map.addLayer({
        id: 'rivers',
        type: 'line',
        source: 'assam-rivers',
        layout: { 'line-cap': 'round', 'line-join': 'round' },
        paint: {
          'line-color': [
            'case',
            ['boolean', ['get', 'major'], false],
            RIVER_COLOR_MAJOR,
            RIVER_COLOR_TRIBUTARY,
          ],
          'line-width': ['case', ['boolean', ['get', 'major'], false], 2.4, 1.1],
          'line-opacity': 0.85,
        },
      })

      // Off by default (layout visibility 'none'), toggled from the
      // Layers panel -- see toggleLayer, which flips this exact property.
      // Never fetched/added conditionally: the geojson request is small
      // (it's a static file, not a live query) and district clicks/hover
      // above already establish the pattern of adding every real layer up
      // front rather than lazily on first toggle.
      map.addLayer({
        id: 'flood-extent-fill',
        type: 'fill',
        source: 'assam-flood-extent',
        layout: { visibility: 'none' },
        paint: { 'fill-color': FLOOD_EXTENT_FILL_COLOR, 'fill-opacity': 0.55 },
      })

      let hoveredDistrict: string | number | null = null
      map.on('mousemove', 'district-fill', (event) => {
        const feature = event.features?.[0]
        if (feature?.id === undefined || feature.id === hoveredDistrict) {
          return
        }
        if (hoveredDistrict !== null) {
          map.setFeatureState({ source: 'assam-districts', id: hoveredDistrict }, { hover: false })
        }
        hoveredDistrict = feature.id
        map.setFeatureState({ source: 'assam-districts', id: feature.id }, { hover: true })
        map.getCanvas().style.cursor = 'pointer'
      })

      map.on('mouseleave', 'district-fill', () => {
        if (hoveredDistrict !== null) {
          map.setFeatureState({ source: 'assam-districts', id: hoveredDistrict }, { hover: false })
          hoveredDistrict = null
        }
        map.getCanvas().style.cursor = ''
      })

      map.on('click', 'district-fill', (event) => {
        const feature = event.features?.[0]
        if (feature?.id === undefined) {
          return
        }
        if (selectedDistrictRef.current !== null) {
          map.setFeatureState(
            { source: 'assam-districts', id: selectedDistrictRef.current },
            { selected: false },
          )
        }
        selectedDistrictRef.current = feature.id
        map.setFeatureState({ source: 'assam-districts', id: feature.id }, { selected: true })
        setSelectedDistrictName(String(feature.properties?.name ?? ''))
      })

      // Real severity coloring, for the districts a real result actually
      // covers -- see lib/assamFloodDemo.ts's own module comment for
      // exactly what "real" means here (a genuine Sentinel-1 pair over one
      // small AOI, not full-state coverage). Applied as a setPaintProperty
      // *after* the initial addLayer above, once the fetch resolves,
      // rather than blocking the map's first paint on a network request.
      fetchAssamFloodDemo().then((demo) => {
        if (!demo) return
        // Prefer the full per-district breakdown (a statewide build) so
        // every genuinely-covered district gets colored, not just the top
        // 5 worst_affected -- summarize()'s 5-item limit is the right
        // contract for a grounded LLM tool result, but not for coloring a
        // map where a district covered at a real, low, non-ranking
        // percentage should still read as "checked" rather than "no data".
        const covered = demo.districts
          ? demo.districts.filter((d) => d.tiles_covering > 0)
          : demo.worst_affected
        if (covered.length === 0) return
        setFloodDemo(demo)

        const maxPercent = Math.max(...covered.map((d) => d.flooded_percent))
        const severityPairs = covered.flatMap((d) => [
          d.name,
          severityColor(maxPercent > 0 ? d.flooded_percent / maxPercent : 0),
        ])

        // MapLibre's ExpressionSpecification type for 'match' expects a
        // fixed tuple shape (label, output, ...more pairs, fallback) it
        // can't infer from a spread built at runtime -- the cast is for
        // the type checker only, the actual array shape is exactly what
        // 'match' expects and is exercised directly in the browser.
        const severityMatchExpression = [
          'match',
          ['get', 'name'],
          ...severityPairs,
          DISTRICT_FILL_COLOR,
        ] as unknown as ExpressionSpecification
        const fillColorExpression: ExpressionSpecification = [
          'case',
          ['boolean', ['feature-state', 'selected'], false],
          DISTRICT_SELECTED_COLOR,
          ['boolean', ['feature-state', 'hover'], false],
          DISTRICT_HOVER_COLOR,
          severityMatchExpression,
        ]
        map.setPaintProperty('district-fill', 'fill-color', fillColorExpression)
      })
    })

    mapRef.current = map

    // StrictMode double-invokes this effect in development, and React 19
    // resets refs between those invocations -- so the mapRef guard above
    // does NOT stop a second map being built. Without this cleanup two
    // MapLibre instances end up alive on the same container, competing
    // for the WebGL context, which is what made the map take ~10 seconds
    // to appear. Tearing the old one down is what keeps exactly one.
    return () => {
      map.remove()
      mapRef.current = null
    }
  }, [])

  function handleBack() {
    const map = mapRef.current
    if (map === null) {
      return
    }
    if (selectedDistrictRef.current !== null) {
      map.setFeatureState(
        { source: 'assam-districts', id: selectedDistrictRef.current },
        { selected: false },
      )
      selectedDistrictRef.current = null
    }
    setSelectedDistrictName(null)
    map.fitBounds(ASSAM_BOUNDS, { ...ASSAM_FIT_OPTIONS, duration: 800 })
  }

  return (
    <div className="map-view-wrapper">
      <div ref={containerRef} className="map-view" data-testid="map-view" />
      {selectedDistrictName !== null && (
        <div className="map-nav">
          <button type="button" className="back-to-india" onClick={handleBack}>
            ← Assam
          </button>
          <span className="current-location">{selectedDistrictName}</span>
        </div>
      )}
      <LayerToggles checked={layersChecked} onToggle={toggleLayer} />
      <SeverityLegend />
      {floodDemo && <AssamFloodDemoBadge demo={floodDemo} />}
    </div>
  )
}
