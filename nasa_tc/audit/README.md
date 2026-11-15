# NASA Tropical Storm distance-to-land

No per-row distance CSV is shipped here.

The NASA loader on main (`src.nasa_tc`) and the official TorchGeo
feature tables it reads (`Image ID`, `Storm ID`, `Relative Time`,
`Ocean`, plus `Wind Speed` on the labels file) do not have a usable
storm id or time for an IBTrACS join:

- `storm_id` is a 3-letter competition identifier (for example `nhe`),
  not an IBTrACS `SID` or a `USA_ATCF_ID`.
- The only time field is `relative_time_s`, the offset in seconds from
  the first frame of that storm. There is no absolute timestamp, and
  no latitude or longitude.

Joining `DIST2LAND` would mean inventing both a mapping from those
3-letter ids to IBTrACS storms and an epoch for `relative_time_s`.
This audit does not guess. TCIR, which does have a storm id and an
absolute time, is in `tcir/audit/distance_to_land.csv`.
