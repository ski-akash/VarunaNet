import { useEffect, useRef, useState } from 'react'
import { Map as MapLibreMap, NavigationControl, type StyleSpecification } from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
import FloodReportBadge from './FloodReportBadge'
import LayerToggles from './LayerToggles'
import SeverityLegend from './SeverityLegend'

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
        compact: true,
        customAttribution:
          'Boundaries: <a href="https://github.com/datameet/maps" target="_blank" rel="noreferrer">DataMeet</a>, Census 2011 (CC BY 4.0)',
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
      <LayerToggles />
      <SeverityLegend />
      <FloodReportBadge />
    </div>
  )
}
