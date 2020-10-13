# Matched Field Processing (MFP)

[![DOI](https://zenodo.org/badge/268340026.svg)](https://zenodo.org/badge/latestdoi/268340026)

Determine nearby sources of seismic signal from dense array data using "Matched Field Processing". The plane-wave-assumption of classical beamforming is violated in cases where sources are near or inside the seismic array. This method is essentially 3D-beamforming, where a grid-search is performed on a 3D-grid (here x,y,z) instead of 2D in classical beamforming (usually backazimuth and velocity). Green's Function spectra are computed for a medium with constant velocity at each grid point and compared to recorded (i.e., synthetic for now) spectra. The beampower is computed using the Bartlett processor.

For now, this is a synthetic demonstration. Eventually, I will expand this to real data.

This code follows the methodology detailed in Umlauft & Korn (2019).

> Umlauft, J., & Korn, M. (2019). 3-D fluid channel location from noise tremors using matched field processing. Geophysical Journal International, 219(3), 1550–1561. [doi](http://doi.org/10.1093/gji/ggz385)

## requirements

- tqdm
