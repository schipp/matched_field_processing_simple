import numpy as np
from tqdm import tqdm
# frequency band to investigate
fmin = 0.2
fmax = 1

# plotting
voxel_confidence = .99

# geometry / structure
grid_limits = (-2, 2)
n_gridpoints_x = 100
n_gridpoints_y = 100
n_gridpoints_z = 10
vel = .3

x, y, z = np.indices((n_gridpoints_x ,n_gridpoints_y ,n_gridpoints_z))
#
#  decimals to round distances to
# should probably be dependent on spatial nyquist?
decimal_round = 2

# synth example
n_stations = 10
synth_source = np.random.uniform(low=grid_limits[0], high=grid_limits[1], size=2)
synth_source = np.append(synth_source, np.random.uniform(low=2, high=4))

# data
n_samples = 100
sampling_rate = 1
# -- go

def get_gf_spectrum(freqs:np.ndarray, dist:float, vel:float) -> np.ndarray:
    """
    Compute the Green's Function (GF) spectrum for given frequencies, distance, and medium-velocity.
    Replace by more complex GFs if 1D, 2D or 3D model is available.
    """
    return np.exp(-1j * freqs * dist/vel)

def get_distances(list_of_locs:np.ndarray, point:np.ndarray) -> np.ndarray:
    """
    Compute the distance between a list of coordinate-pairs and a single point.
    If you want 3D distances (source at depth) list_of_locs = [(lat1, lon1, alt1), ...]
    """
    return np.linalg.norm(list_of_locs - point, ord=2, axis=1)

# compute synthetic station locations
sta_lons = np.random.uniform(low=grid_limits[0], high=grid_limits[1], size=n_stations)
sta_lats = np.random.uniform(low=grid_limits[0], high=grid_limits[1], size=n_stations)
sta_alts = np.zeros(n_stations)
station_locations = np.array((sta_lons, sta_lats, sta_alts)).T

# compute fft frequencies
from scipy.fftpack import fftfreq
freqs = fftfreq(n_samples, sampling_rate)
freqs_of_interest = (freqs >= fmin) & (freqs <= fmax)

# compute synthetic spectra
dists_to_src = get_distances(list_of_locs=station_locations, point=synth_source)
source_spectra = np.array([get_gf_spectrum(freqs, dist, vel)[freqs_of_interest] for dist in dists_to_src])

# grid geometry
from itertools import product
grid_x_coords = np.linspace(grid_limits[0], grid_limits[1], n_gridpoints_x)
grid_y_coords = np.linspace(grid_limits[0], grid_limits[1], n_gridpoints_y)
grid_z_coords = np.linspace(0, 4, n_gridpoints_z)
grid_points = np.asarray(list(product(grid_x_coords, grid_y_coords, grid_z_coords)))

grid_x_coords_edge = np.append(grid_x_coords, grid_limits[1] + (grid_limits[1] - grid_limits[0]) / n_gridpoints_x)
grid_y_coords_edge = np.append(grid_y_coords, grid_limits[1] + (grid_limits[1] - grid_limits[0]) / n_gridpoints_y)
grid_z_coords_edge = np.append(grid_z_coords, 4 + (4 - 0) / n_gridpoints_z)

# compute cross-spectral density-matrix (csdm)
csdm = np.zeros((n_stations, n_stations, len(freqs[freqs_of_interest])), dtype=np.complex)
for i, spec1 in enumerate(source_spectra):
    for j, spec2 in enumerate(source_spectra):
        csdm[i, j] = spec1 * np.conj(spec2)

# compute all distances
gp_dists = []
for gp in grid_points:
    dists = get_distances(list_of_locs=station_locations, point=gp)
    # round to deciamsl to reduce number of required synth spectra
    dists = np.round(dists, decimals=decimal_round)
    gp_dists.append(dists)

# compute green's functions spectra only for relevant dists
relevant_dists = np.unique(gp_dists)
gf_spectra = np.array([get_gf_spectrum(freqs, dist, vel)[freqs_of_interest] for dist in relevant_dists])

beampowers_at_gps = []
for gp in tqdm(grid_points):
    a = np.round(np.linalg.norm(station_locations - gp, ord=2, axis=1), decimals=2)
    B_as = []
    for i, sta1 in enumerate(station_locations):
        for j, sta2 in enumerate(station_locations):
            # extract pre-computed spectra
            synth_spectrum_i_conj = np.conj(gf_spectra[relevant_dists==a[i]])
            synth_spectrum_j = gf_spectra[relevant_dists==a[j]]
            # Bartlett processor
            B_ai_aj = np.sum(np.real(synth_spectrum_i_conj * csdm[i, j] * synth_spectrum_j))
            B_as.append(B_ai_aj)
    beampowers_at_gps.append(np.sum(B_as))

# voxel plot
voxels = np.array(beampowers_at_gps).reshape(n_gridpoints_x ,n_gridpoints_y ,n_gridpoints_z)
voxels = voxels >= voxel_confidence*np.max(np.abs(beampowers_at_gps))

colors = np.empty(voxels.shape, dtype=object)
voxels = np.transpose(voxels, (1, 0, 2))
colors[voxels] = 'orange'

# plot example
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
# fig, ax = plt.subplots(1, 1, projection='3d')
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
# ax.set_aspect('equal')

# ax = fig.gca(projection='3d')
# xx, yy, zz = np.meshgrid(grid_x_coords, grid_y_coords, grid_z_coords)
xx, yy, zz = np.meshgrid(grid_x_coords_edge, grid_y_coords_edge, grid_z_coords_edge)
ax.voxels(xx, yy, zz, voxels, facecolors=colors, edgecolor='grey', alpha=1, lw=.25)

ax.plot([synth_source[0], synth_source[0]], [synth_source[1], synth_source[1]], [4, synth_source[2]], c='k', lw=.5)
ax.scatter(synth_source[0], synth_source[1], synth_source[2])
ax.scatter(station_locations[:, 0], station_locations[:, 1], station_locations[:, 2], c='k', marker="^", edgecolors='w')

ax.set_title(f"{synth_source}")

# plot 0-depth slice at surface
beampowers_at_gps = np.array(beampowers_at_gps).reshape(n_gridpoints_x, n_gridpoints_y, n_gridpoints_z)
vmin, vmax = np.min(beampowers_at_gps), np.max(beampowers_at_gps)
data = beampowers_at_gps[:, :, 0]
xx, yy = np.meshgrid(grid_x_coords, grid_y_coords)
ax.contourf(xx, yy, data.T, 100, zdir='z', offset=grid_z_coords[0], cmap='magma', alpha=1, vmin=vmin, vmax=vmax)

ax.set_xlim(grid_limits[0], grid_limits[1])
ax.set_ylim(grid_limits[0], grid_limits[1])
ax.set_zlim(0, 4)
ax.invert_zaxis()

plt.show(fig)
plt.close(fig)