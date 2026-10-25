# Data sources

Do not commit imagery or best-track dumps. CI must not download them.

- NASA Tropical Storm Wind Speed, TorchGeo `TropicalCyclone`.
  Atlantic and East Pacific, 2000–2019. Download uses azcopy.
  Cite https://doi.org/10.1109/JSTARS.2020.3011907.
  The Hugging Face mirror `torchgeo/tropical_cyclone` lists CC-BY-4.0 and
  about 2.4 GB. Do not depend on the Hub viewer in CI. A local copy can
  be turned into a manifest later; this phase does not ship a downloader.
- TCIR: https://github.com/BoyoChen/TCIR — MIT repository, 201×201, four
  channels. Confirm satellite redistribution terms separately. Southern
  Hemisphere storms are not silently flipped. Phase 2 features use the
  first channel only.
- IBTrACS v4: https://doi.org/10.25921/82ty-9e16. Labels only.
- INSAT-3D / 3DR: MOSDAC, registration required, not redistributed here.
  The intensity page reads the public Infrared-1 JPEG on
  https://mausam.imd.gov.in/responsive/satellite.php when someone clicks
  it. That file is not committed, and tests do not download it.
- TC PRIMED: https://doi.org/10.25921/dmy1-0595. Optional later source.

`wind_averaging` must be a column on every manifest, for example `1-min`,
`3-min`, or `10-min`. Mixed periods in one manifest are rejected. The
17/34/48/64/91 kt names are this project's grouping, not official IMD classes,
and they are not a warning product.

`center_source` may be `dataset_precentered` or a fix available at image
time. Do not crop with a best-track center revised after the image time.
Phase 2 radial and eye features assume the crop is already centered.

Smoke manifests from `src/make_smoke_data.py` use `wind_averaging=synthetic`
and `is_synthetic=true`. They are not a dataset.
