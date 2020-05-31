# Matched Field Processing

Determine continuous sources of seismic signal from dense array data.

## required maths/physics knowledge

### Cross-spectral density matrix(CSDM)

![equation](http://www.sciweavers.org/tex2img.php?eq=K_%7Bij%7D%28%5Comega%29%20%3D%20d_i%28%5Comega%29%20d_j%5E%2A%28%5Comega%29&bc=White&fc=Black&im=png&fs=12&ff=arev&edit=0)

### Greens's Functions (1D)

for array element *i*, distance *a*, phase velocity *c*

Body waves ![equation](http://www.sciweavers.org/tex2img.php?eq=%5Chat%7Bd_i%7D%28%5Comega%2C%20a%29%20%3D%20%5Cfrac%7B1%7D%7B4%5Cpi%20a%7D%5Cexp%5Cleft%28%5Cfrac%7B-i%5Comega%20a%7D%7Bc%7D%5Cright%29&bc=White&fc=Black&im=png&fs=12&ff=arev&edit=0)

Surface waves ![equation](http://www.sciweavers.org/tex2img.php?eq=%5Chat%7Bd_i%7D%28%5Comega%2C%20a%29%20%3D%20%5Csqrt%7B%5Cfrac%7B2%7D%7B%5Cpi%20a%7D%7D%5Cexp%5Cleft%28%5Cfrac%7B-i%5Cpi%7D%7B4%7D%5Cright%29%5Cexp%5Cleft%28%5Cfrac%7B-i%5Comega%20a%7D%7Bc%7D%5Cright%29&bc=White&fc=Black&im=png&fs=12&ff=arev&edit=0)

Amplitude terms can be dropped (unreliable) -> Normalize seismic records.

### Beampower estimator

Bartlett processor ![equation](http://www.sciweavers.org/tex2img.php?eq=B_%7BBart%7D%28a%29%20%3D%20%5Csum_%7B%5Comega%7D%7C%5Chat%7Bd_i%5E%2A%7D%28%5Comega%2C%20a%29%20K_%7Bij%7D%20%28%5Comega%29%20%5Chat%7Bd_j%5E%2A%7D%28%5Comega%2C%20a%29%7C&bc=White&fc=Black&im=png&fs=12&ff=arev&edit=0)

## program logic

1. read seismic data & metadata
2. treat traces (normalize, write lat/lon into tr.stats.coordinates)
3. compute CSDM K for frequencies of interest
4. define grid
    a. compute (rounded to nyquist?) distances from stations to grid points
5. for each distance compute synthetic waveform (can reduce required computations if using approx.)
6. for each grid point
    a. for each frequency of interest
        - compute Bartlett processor: is this possible as matrix multiplication, automatically parallelized? number of station n_stations, length of traces n_samples
          - shape of arrays: (n_stations, n_samples) x (n_stations, n_stations, n_samples) x (n_stations, n_samples)
          - example: (20, 1000) x (20, 20, 1000) x (20, 1000)
        - sum individual bartlett processors of frequencies
