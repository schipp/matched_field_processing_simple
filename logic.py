import numpy as np
from tqdm import tqdm

def get_gf_spectrum(freqs:np.ndarray, dist:float, vel:float) -> np.ndarray:
    """
    Compute the Green's Function (GF) spectrum for given frequencies, distance, and medium-velocity.
    Replace by more complex GFs if 1D, 2D or 3D model is available.
    """
    return np.exp(-1j * freqs * dist/vel)

def get_distances(list_of_locs:np.ndarray, point:np.ndarray) -> np.ndarray:
    """
    Compute the distance between an array of coordinate-pairs and a single point.
    If you want 3D distances (source at depth) list_of_locs = [(lat1, lon1, alt1), ...]
    """
    return np.linalg.norm(list_of_locs - point, ord=2, axis=1)

def generate_grid(grid_limits, n_gridpoints):
    """
    Generate the grid geometry
    Returns coordinates of n grid_points as [[x0,y0,z0], [x1,y1,z1], ..., [xn,yn,zn]|
    """

    grid_limits_x, grid_limits_y, grid_limits_z = grid_limits
    n_gridpoints_x, n_gridpoints_y, n_gridpoints_z = n_gridpoints
    
    # grid geometry
    from itertools import product
    grid_x_coords = np.linspace(grid_limits_x[0], grid_limits_x[1], n_gridpoints_x)
    grid_y_coords = np.linspace(grid_limits_y[0], grid_limits_y[1], n_gridpoints_y)
    grid_z_coords = np.linspace(grid_limits_z[0], grid_limits_z[1], n_gridpoints_z)
    grid_points = np.asarray(list(product(grid_x_coords, grid_y_coords, grid_z_coords)))
    return grid_points, grid_x_coords, grid_y_coords, grid_z_coords

def get_csdm(spectra:np.ndarray) -> np.ndarray:
    """
    Compute the cross-spectral density-matrix (CSDM)
    spectra: (N x M) shaped np.ndarray with N recorded spectra of length M
    """
    
    # einstein convention version
    csdm = np.einsum('ik,jk->ijk', source_spectra, np.conj(source_spectra))

    # # old slow version
    # csdm = np.zeros((n_stations, n_stations, len(freqs[freqs_of_interest])), dtype=np.complex)
    # for i, spec1 in enumerate(source_spectra):
    #     for j, spec2 in enumerate(source_spectra):
    #         csdm[i, j] = spec1 * np.conj(spec2)

    return csdm

def get_all_distances_rounded(station_locations, grid_points, decimal_round) -> list:
    """
    Computes the distances between all station locations and grid_points.
    """

    gp_dists = []
    for gp in grid_points:
        dists = get_distances(list_of_locs=station_locations, point=gp)
        # round to deciamsl to reduce number of required synth spectra
        dists = np.round(dists, decimals=decimal_round)
        gp_dists.append(dists)
    
    return gp_dists


def get_gf_spectra_for_dists(freqs, vel, dists) -> np.ndarray:
    """
    Computes the Green's Functions spectra for relevant distances only
    """

    gf_spectra = np.array([get_gf_spectrum(freqs, dist, vel) for dist in dists])
    
    return gf_spectra


def bartlett_processor(csdm, gf_spectra):
    """
    Computes the beampower using the Bartlett Processor
    """

    beampower = np.real(np.einsum('ik,ijk,jk', np.conj(gf_spectra), csdm, gf_spectra))

    return beampower

def get_beampowers(csdm, gf_spectra, gp_dists):
    """
    Computes the Beampower for all grid-points.
    """

    beampowers = []
    for dists in tqdm(gp_dists):
        # dists = get_distances(list_of_locs=station_locations, point=gp)
        # dists = np.round(dists, decimals=decimal_round)
        
        dist_idxs = np.array([np.where(np.isin(relevant_dists, dist)) for dist in dists]).flatten()
        gf_spectra_relevant = gf_spectra[dist_idxs]

        beampower = bartlett_processor(csdm=csdm, gf_spectra=gf_spectra_relevant)

        beampowers.append(beampower)
    
    return beampowers



if __name__ == '__main__':
    # frequency band to investigate
    fmin = 0.2
    fmax = 1

    # plotting
    voxel_confidence = .99

    # geometry / structure
    # TODO: determine lower bounds for grid geometry
    grid_limits_x = (-2, 2)
    grid_limits_y = (-2, 2)
    grid_limits_z = (0, 4)
    n_gridpoints_x = 40
    n_gridpoints_y = 40
    n_gridpoints_z = 10
    vel = .1

    # decimals to round distances to
    # TODO: Dynamically determine maximum reasonable accuracy
    # should probably be dependent on spatial nyquist?
    decimal_round = 2

    # data
    n_samples = 1000
    sampling_rate = 1
    # -- go

    # synth example
    n_stations = 30
    synth_source_x = np.random.uniform(low=grid_limits_x[0], high=grid_limits_x[1])
    synth_source_y = np.random.uniform(low=grid_limits_y[0], high=grid_limits_y[1])
    synth_source_z = np.random.uniform(low=2, high=grid_limits_z[1])
    synth_source = np.array((synth_source_x, synth_source_y, synth_source_z))

    # compute synthetic station locations
    sta_lons = np.random.uniform(low=grid_limits_x[0], high=grid_limits_x[1], size=n_stations)
    sta_lats = np.random.uniform(low=grid_limits_y[0], high=grid_limits_y[1], size=n_stations)
    sta_alts = np.zeros(n_stations)
    station_locations = np.array((sta_lons, sta_lats, sta_alts)).T

    # compute fft frequencies
    from scipy.fftpack import fftfreq
    freqs = fftfreq(n_samples, sampling_rate)
    freqs_of_interest = (freqs >= fmin) & (freqs <= fmax)

    # compute synthetic spectra
    dists_to_src = get_distances(list_of_locs=station_locations, point=synth_source)
    source_spectra = np.array([get_gf_spectrum(freqs, dist, vel)[freqs_of_interest] for dist in dists_to_src])

    grid_points, grid_x_coords, grid_y_coords, grid_z_coords = generate_grid(
        grid_limits=(grid_limits_x, grid_limits_y, grid_limits_z), 
        n_gridpoints=(n_gridpoints_x, n_gridpoints_y, n_gridpoints_z)
    )
    
    csdm = get_csdm(
        spectra=source_spectra
        )

    # compute all station-gridpoint-distances, rounded
    gp_dists = get_all_distances_rounded(
        station_locations=station_locations,
        grid_points=grid_points,
        decimal_round=decimal_round
        )

    # extract relevant distances only
    relevant_dists = np.unique(gp_dists)

    # compute Green's Functions (spectra)
    gf_spectra = get_gf_spectra_for_dists(
        freqs=freqs,
        vel=vel,
        dists=relevant_dists
        )

    # limit Green's Functions spectra to frequencies of interest
    gf_spectra = gf_spectra[:, freqs_of_interest]

    # compute beampowers
    beampowers = get_beampowers(
        csdm=csdm,
        gf_spectra=gf_spectra,
        gp_dists=gp_dists
        )

    # np.argmax(beampowers)

    # plot example
    import pylab as plt
    from mpl_toolkits.mplot3d import Axes3D
    # fig, ax = plt.subplots(1, 1, projection='3d')
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    # prepare voxels for voxel plot
    beampowers = np.array(beampowers).reshape(n_gridpoints_x, n_gridpoints_y, n_gridpoints_z)
    voxels = beampowers >= voxel_confidence*np.max(np.abs(beampowers))

    colors = np.empty(voxels.shape, dtype=object)
    voxels = np.transpose(voxels, (1, 0, 2))
    colors[voxels] = 'orange'
    # add edge coordinates for voxel plotting
    gridsize_x = (grid_limits_x[1] - grid_limits_x[0]) / n_gridpoints_x
    gridsize_y = (grid_limits_y[1] - grid_limits_y[0]) / n_gridpoints_y
    gridsize_z = (grid_limits_z[1] - grid_limits_z[0]) / n_gridpoints_z
    grid_x_coords_edge = np.append(grid_x_coords, grid_limits_x[1] + gridsize_x) - gridsize_x/2
    grid_y_coords_edge = np.append(grid_y_coords, grid_limits_y[1] + gridsize_y) - gridsize_y/2
    grid_z_coords_edge = np.append(grid_z_coords, grid_limits_z[1] + gridsize_z) - gridsize_z/2
    # meshgrid for voxels
    xx, yy, zz = np.meshgrid(grid_x_coords_edge, grid_y_coords_edge, grid_z_coords_edge)
    ax.voxels(xx, yy, zz, voxels, facecolors=colors, edgecolor='grey', alpha=1, lw=.25)

    ax.plot([synth_source[0], synth_source[0]], [synth_source[1], synth_source[1]], [4, synth_source[2]], c='k', lw=.5)
    ax.scatter(synth_source[0], synth_source[1], synth_source[2])
    ax.scatter(station_locations[:, 0], station_locations[:, 1], station_locations[:, 2], c='k', marker="^", edgecolors='w')

    ax.set_title(f"synthetic source at: {synth_source[0]:0.2f}, {synth_source[1]:0.2f}, {synth_source[2]:0.2f}\nbest fit at: {grid_points[np.argmax(beampowers)][0]:0.2f}, {grid_points[np.argmax(beampowers)][1]:0.2f}, {grid_points[np.argmax(beampowers)][2]:0.2f}")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # plot 0-depth slice at surface
    vmin, vmax = np.min(beampowers), np.max(beampowers)
    data = beampowers[:, :, 0]
    xx, yy = np.meshgrid(grid_x_coords, grid_y_coords)
    ax.contourf(xx, yy, data.T, 100, zdir='z', offset=grid_z_coords[0], cmap='magma', alpha=1, vmin=vmin, vmax=vmax, zorder=10)

    ax.set_xlim(grid_limits_x[0], grid_limits_x[1])
    ax.set_ylim(grid_limits_y[0], grid_limits_y[1])
    ax.set_zlim(grid_limits_z[0], grid_limits_z[1])
    ax.invert_zaxis()

    plt.show(fig)
    plt.close(fig)