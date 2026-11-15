# TCIR distance-to-land flags

Per-row distance to nearest land for the TCIR manifest, joined from
IBTrACS v04 `DIST2LAND`. This is the column name Clara's PR #35 looks
for. Open water is a distance greater than 20 km. `src/` is not
modified; the loader can join this CSV later.

NASA Tropical Storm rows are **not** in this file. The NASA loader on
main only has a 3-letter competition `storm_id` (for example `nhe`) and
`relative_time_s`. It has no absolute timestamp and no IBTrACS or ATCF
id, so those rows cannot be joined without guessing. See
`nasa_tc/audit/README.md`.

## Source

- Official NOAA NCEI IBTrACS v04r01 CSV, subset `since1980` (covers
  every TCIR year, 2003–2016):
  [ibtracs.since1980.list.v04r01.csv](https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.since1980.list.v04r01.csv)
- Version: **v04r01**. DOI: [10.25921/82ty-9e16](https://doi.org/10.25921/82ty-9e16).
- Retrieved 16 November 2026, 12:20 AM IST, from that NCEI URL. The
  listing on that directory showed a last-modified time of 12 November
  2026, 10:06 (the timestamp NCEI posted with the file).
- MD5 of the retrieved file: `10b051164e89472fa392b8f48170257b`.
- The IBTrACS dump itself is not committed.

`DIST2LAND` is kilometres to nearest land. IBTrACS documents that a
value of 0 means landfall within 6 hours.

## Join rule

The committed CSV is keyed by `sample_id`, the same key `src.tcir`
writes on the manifest (for example `tcir_200301L_2003041800`).
Columns: `sample_id`, `distance_to_land_km`, `near_land`.

TCIR `storm_id` values have the form `YYYYNNX`. That is rewritten to
the IBTrACS `USA_ATCF_ID` `PPNNYYYY` with basin letters L=AL, E=EP,
W=WP, C=CP, I=IO, S=SH. Example: `200301L` → `AL012003`. The join is
exact on that id and on `ISO_TIME` (0-second tolerance). TCIR times are
already on the 3-hour UTC marks IBTrACS uses. There is no interpolation
and no fallback on latitude or longitude.

`near_land` is:

- `true` if `distance_to_land_km` ≤ 20 (including IBTrACS 0)
- `false` if `distance_to_land_km` > 20
- `unknown`, with a blank distance, when no IBTrACS row matches that
  id and time

## Counts in the committed file

69,867 TCIR rows (the full `src.tcir` manifest):

| near_land | rows |
| --- | --- |
| true | 5,027 |
| false | 64,768 |
| unknown | 72 |

69,795 rows (99.9%) match an IBTrACS fix. The 72 unknowns are left
blank on purpose:

- `201622E` (constructed `EP222016`): all 71 frames. That ATCF id is
  not in the retrieved IBTrACS file.
- `201011E` (`EP112010`, Hermine): 1 frame at 2010-09-04 18:00 UTC.
  The IBTrACS track for that id ends at 15:00 UTC the same day.

## Regenerating

Audit-only dependency: `pandas`. Not part of CI.

```bash
python tcir/audit/build_distance_to_land.py \
  --manifest /path/to/tcir/manifest.csv \
  --ibtracs /path/to/ibtracs.since1980.list.v04r01.csv \
  --output tcir/audit/distance_to_land.csv
```
