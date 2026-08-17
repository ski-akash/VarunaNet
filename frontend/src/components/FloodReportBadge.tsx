import { CURRENT_FLOOD_REPORT } from '../lib/currentFloodReports'

// Small attribution badge for the reported-flood-affected districts
// highlighted on the map -- keeps the source and date visible right next
// to the data, not buried in a code comment nobody using the dashboard
// would ever see. The unmatched-districts note matters here too: a
// district silently missing (rather than explained) reads as the map
// being wrong, not as a known, deliberate data limitation.
export default function FloodReportBadge() {
  return (
    <div className="flood-report-badge">
      <span>
        Reported flood-affected districts as of {CURRENT_FLOOD_REPORT.asOf} — from news reports,
        not SAR-measured extent.{' '}
        <a href={CURRENT_FLOOD_REPORT.sourceUrl} target="_blank" rel="noreferrer">
          Source
        </a>
      </span>
      {CURRENT_FLOOD_REPORT.unmatchedDistricts.length > 0 && (
        <span className="flood-report-badge-note">
          Also reported affected but not shown (created after the 2011 census boundaries this map
          uses): {CURRENT_FLOOD_REPORT.unmatchedDistricts.join(', ')}.
        </span>
      )}
    </div>
  )
}
