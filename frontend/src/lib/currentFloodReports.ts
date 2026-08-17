// Real, sourced, current flood reporting -- NOT SAR-measured flood
// extent. This project's whole point is eventually replacing exactly
// this kind of news-report data with actual model output (pixel-level
// SAR-derived masks, spec section 1), so it matters a lot that this
// stays clearly distinguished from that: no km² figures, no severity
// gradient, no confidence score -- just "reported as flood-affected by
// news sources as of a specific date," shown with a visibly different
// style (see MapView's reported-flood-affected layer) from the
// severity-legend gradient built for real model output.
//
// District names matched exactly against the 2011 Census district
// dataset (public/geo/india_districts.geojson) already in use. Two
// reported-affected districts could NOT be matched and are deliberately
// left out rather than approximated onto a parent district:
//   - Charaideo (created 2015-16, carved out of Sivasagar)
//   - Biswanath (created 2015, carved out of Sonitpur)
// Both postdate the 2011 census boundaries this map uses. Silently
// coloring the old, larger Sivasagar/Sonitpur to stand in for their
// newer, smaller successor districts would overstate the affected area
// -- so instead they're named explicitly as "not shown" wherever this
// data gets displayed, rather than either silently dropped or silently
// misattributed.
export const CURRENT_FLOOD_REPORT = {
  asOf: '2026-08-15',
  affectedDistricts: [
    'Darrang',
    'Golaghat',
    'Jorhat',
    'Karbi Anglong',
    'Nagaon',
    'Sivasagar',
    'Kamrup Metropolitan',
  ],
  unmatchedDistricts: ['Charaideo', 'Biswanath'],
  sourceLabel: '2026 Assam floods (Wikipedia / news reports)',
  sourceUrl: 'https://en.wikipedia.org/wiki/2026_Assam_floods',
} as const
