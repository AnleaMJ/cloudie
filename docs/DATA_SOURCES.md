# Data sources

Do not commit imagery, best-track dumps, or model weights. `data/*`, `*.h5`, `*.pt`, and `*.ckpt` are gitignored. The notes below are for a later download step on a machine that is allowed to store the files. Confirm each provider's terms again before copying pixels.

## NASA Tropical Storm Wind Speed

TorchGeo class: `torchgeo.datasets.TropicalCyclone`.

Atlantic and East Pacific tropical storms from 2000 to 2019, with maximum sustained surface wind speed. Native image size in TorchGeo is 366. The collection was built for the DrivenData competition [Predict Wind Speeds of Tropical Storms](https://www.drivendata.org/competitions/72/predict-wind-speeds/).

- License on the Source Cooperative card: [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). Card: https://data.source.coop/nasa/tropical-storm-competition/README.md
- Paper TorchGeo asks you to cite: https://doi.org/10.1109/JSTARS.2020.3011907 (Maskey et al., Deepti).
- Dataset DOI on that card: https://doi.org/10.34911/rdnt.xs53up
- Download path: TorchGeo shells out to `azcopy` against Source Cooperative. Phase 0–1 does not vendor a downloader, and CI must not download this archive.

A Hugging Face copy lives at https://huggingface.co/datasets/torchgeo/tropical_cyclone. As checked for this commit, the Hub dataset viewer fails with `DatasetGenerationError` and is not a usable table. Do not call `datasets.load_dataset` on it from CI.

The competition publishes a train/test cut by frame. Cloudie re-splits whatever labeled archive you load by storm (`storm_wise`, `chronological`, or `holdout_basin`). Do not treat the competition frame split as leak-free, and do not tune on a final holdout.

## TCIR

Repository: https://github.com/BoyoChen/TCIR (MIT). Description and HDF5 layout: https://www.csie.ntu.edu.tw/~htlin/program/TCIR/

Frames are 201×201 with four channels: IR1 (infrared), water vapour, visible, and passive microwave. IR1 is the channel this project would read first. Missing values are stored as NaN. The passive-microwave channel is interpolated from a coarser CMORPH grid.

Satellite pixels come from GridSat and CMORPH. Those upstream terms are separate from the MIT license on the GitHub repository. The TCIR notes say Southern Hemisphere storms rotate in the other direction. Do not silently flip them.

Keep `TCIR.h5` out of git (`*.h5` is ignored).

## IBTrACS

International Best Track Archive for Climate Stewardship, version 4.

- DOI: https://doi.org/10.25921/82ty-9e16
- Product page: https://www.ncei.noaa.gov/products/international-best-track-archive

Use it as a label join only: position, wind, pressure, and wind radii when the agency provided them. Wind speed, R34, and radius of maximum wind from IBTrACS are in `forbidden_features` for the primary model. NCEI states that IBTrACS follows an open research-distribution policy. Read the current terms on the product page before a bulk copy. "Active" rows are not a forecast.

Record `wind_averaging`. IMD uses a 3-minute maximum sustained wind. NHC and JTWC products are often 1-minute. One manifest may contain one averaging period. Mixed periods fail validation.

## INSAT-3D / INSAT-3DR (later)

MOSDAC (https://www.mosdac.gov.in/) distributes INSAT imagery. Access is registration-gated. This repository assumes redistribution is not allowed. Do not commit INSAT or MOSDAC files. A later North Indian Ocean adapter can convert digital numbers to brightness temperature only when a calibration table is supplied (`cloudie.preprocessing.dn_to_brightness_temperature`). The table is not embedded here.

## TC PRIMED (later)

Tropical Cyclone Precipitation, Infrared, Microwave, and Environmental Dataset.

- DOI: https://doi.org/10.25921/dmy1-0595
- Product notes: https://rammb-data.cira.colostate.edu/tcprimed/products.html

Public NOAA distribution. Not required to boot Phase 0–1. Infrared in TC PRIMED is tied to microwave overpass times, which is a different cadence from a geostationary image sequence.

## Centers and calibration

`center_source` may be `dataset_precentered` or a fix that was available at image time. This phase does not crop using a best-track center that was revised after the image time.

Pixel values are not called brightness temperature unless the caller passes gain and offset. Smoke images are synthetic digital values with a cold core that varies with the fake wind label. They are not calibrated and they are not cyclones.
