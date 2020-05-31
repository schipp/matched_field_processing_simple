import numpy as np
import logging

n_stations = 20
n_samples = 100
sampling_rate = 1
vel = 1

# synth example
sta_lons = np.random.uniform(low=-2, high=2, size=n_stations)
sta_lats = np.random.uniform(low=-2, high=2, size=n_stations)
station_locations = np.array((sta_lons, sta_lats)).T
# random_traces = np.random.random(size=(n_stations, n_samples))
from scipy.fftpack import fftfreq
# random_spectra = np.array([fft(trace, n=n_samples) for trace in random_traces])
# random_spectra_conj = np.conj(random_spectra)

freqs = fftfreq(n_samples, sampling_rate)
from itertools import product
grid_x_coords = np.linspace(-2, 2, 20)
grid_y_coords = np.linspace(-2, 2, 20)
grid_points = np.asarray(list(product(grid_x_coords, grid_y_coords)))
synth_source = np.array((.5, .5))

freqs_of_interest = (freqs > 0.2) & (freqs < 1)

# source spectra
dists_to_src = np.linalg.norm(station_locations - synth_source, ord=2, axis=1)
source_spectra = np.array([np.exp(-1j * freqs * dist/vel)[freqs_of_interest] for dist in dists_to_src])


# compute csdm
csdm = np.zeros((n_stations, n_stations, len(freqs[freqs_of_interest])), dtype=np.complex)
for i, spec1 in enumerate(source_spectra):
    for j, spec2 in enumerate(source_spectra):
        csdm[i, j] = spec1 * np.conj(spec2)

beampowers = []
# beampowers_mm = []
for gp in grid_points:
    a = np.linalg.norm(station_locations - gp, ord=2, axis=1)
    synth_spectra = []
    
    # how to parallelize this?
    # would be great if possible to push into matrix formulation - think tomorrow
    B_as = []
    for i, sta1 in enumerate(station_locations):
        for j, sta2 in enumerate(station_locations):
            synth_spectrum_i = np.exp(-1j * freqs * a[i]/vel)[freqs_of_interest]
            synth_spectrum_j = np.exp(-1j * freqs * a[j]/vel)[freqs_of_interest]
            B_ai_aj = np.sum(np.real(np.conj(synth_spectrum_i) * csdm[i, j] * synth_spectrum_j))
            B_as.append(B_ai_aj)
            # if i != j:
            #     break
    beampowers.append(np.sum(B_as))
    
    # for _ in a:
    #     synth_spectrum_j = np.exp(-1j * freqs * a[j]/vel)[freqs_of_interest]
    #     synth_spectra.append(synth_spectrum)
    #     B_as.append(B_a)
    # # synth_spectra = np.array(synth_spectra)
    # 
    # # matrix multiplication try
    # B = np.real(np.conj(synth_spectra) * csdm * synth_spectra)
    # B = B.reshape(n_stations**2, len(freqs[freqs_of_interest]))
    # B_freq_sum = np.sum(B, axis=1)
    # B_sta_sum = np.sum(B_freq_sum)
    # beampowers.append(B_sta_sum)

    # Bs = []
    # for i, spec1 in enumerate(synth_spectra):
    #     for j, spec2 in enumerate(synth_spectra):
    #         B = np.abs(np.conj(spec1) * csdm[i, j] * spec2)
    #         B_freq_sum = np.sum(B)
    #         Bs.append(B_freq_sum)
    # B_sta_sum = np.sum(Bs)
    # beampowers.append(B_sta_sum)

# plot example
import pylab as plt
fig, ax = plt.subplots(1, 1)
ax.set_aspect('equal')

xx, yy = np.meshgrid(grid_x_coords, grid_y_coords)

pcm = ax.pcolormesh(xx, yy, np.array(beampowers).reshape(len(grid_x_coords), len(grid_y_coords)).T, vmin=np.min(beampowers), vmax=np.max(beampowers))
ax.scatter(sta_lons, sta_lats, c='k', marker='^')
ax.scatter(synth_source[0], synth_source[1], c='red', edgecolors='k', lw=.5)

ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)

x0, y0, w, h = ax.get_position().bounds
cbax = fig.add_axes([x0 + w + .05*w, y0, .05*w, h])
cbar = plt.colorbar(pcm, cax=cbax)

plt.show(fig)
plt.close(fig)