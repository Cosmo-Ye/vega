
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.ndimage import gaussian_filter1d
from scipy.sparse import csr_matrix, identity
from scipy.interpolate import RectBivariateSpline

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from vega.vega_interface import VegaInterface
from vega.plots.wedges import Wedge
import vega.data
from vega.coordinates import Coordinates
import vega.metals

# Apply style
style_path = os.path.join(os.path.dirname(__file__), '5par.mplstyle')
if os.path.exists(style_path):
    plt.style.use(style_path)
else:
    print(f"Warning: Style file {style_path} not found. Using default style.")

class SmoothingMatrix:
    """Matrix-like object that applies Gaussian smoothing along the RP axis (axis 0)."""
    def __init__(self, n_rp, n_rt, sigma_pixels):
        self.n_rp = n_rp
        self.n_rt = n_rt
        self.sigma = sigma_pixels

    def dot(self, x):
        # x is 1D array of size n_rp * n_rt
        # Reshape to (n_rp, n_rt) - see Coordinates.py for layout
        # In Coordinates, rt is X (columns), rp is Y (rows).
        # Flattening is row-major, so we have blocks of constant rp?
        # Wait, let's re-verify Coordinates.py layout:
        # rt_regular_grid, rp_regular_grid = np.meshgrid(rt, rp)
        # default meshgrid('xy'):
        # output[0] (rt_grid) has shape (len(rp), len(rt)) and varies across columns.
        # output[1] (rp_grid) has shape (len(rp), len(rt)) and varies across rows.
        # So row 0 corresponds to rp[0], row 1 to rp[1].
        # Flattening means row 0 comes first.
        # So we have blocks of (fixed rp, varying rt).
        # This means rp varies along AXIS 0 (rows).
        # So we want to smooth along AXIS 0.
        
        try:
            grid = x.reshape((self.n_rp, self.n_rt))
            # Smooth along axis 0 (rp direction)
            # mode='nearest' is usually best for correlation functions at boundaries
            smoothed = gaussian_filter1d(grid, self.sigma, axis=0, mode='nearest')
            return smoothed.flatten()
        except Exception as e:
            print(f"Error in SmoothingMatrix.dot: {e}")
            return x


def parse_args():
    parser = argparse.ArgumentParser(description="Replot Polychord results")
    parser.add_argument('-wedge', action='store_true', help="Plot wedge correlation function")
    parser.add_argument('-compare', action='store_true', help="Plot ratio of two chains")
    parser.add_argument('-one', action='store_true', help="Single panel for compare mode")
    parser.add_argument('-four', action='store_true', help="Four panels for compare mode (default)")
    parser.add_argument('--chain1', type=str, default='Polychord_result/auto_fiducial', help="Path to first chain directory")
    parser.add_argument('--chain2', type=str, default=None, help="Path to second chain directory (for compare)")
    parser.add_argument('--config', type=str, default='analysis/main_fid.ini', help="Path to Vega config file (used for both chains if --config1/--config2 not provided)")
    parser.add_argument('--config1', type=str, default=None, help="Path to Vega config for chain1 (overrides --config for chain1)")
    parser.add_argument('--config2', type=str, default=None, help="Path to Vega config for chain2 (overrides --config for chain2)")
    parser.add_argument('--corr', type=str, default='lyaxlya', help="Correlation name to plot")
    parser.add_argument('--plot-rmin', type=float, default=10.0, help="Model curve r-min for wedge")
    parser.add_argument('--plot-rmax', type=float, default=180.0, help="Model curve r-max for wedge")
    return parser.parse_args()

def read_params(chain_dir):
    """Read parameters from Polychord .stats file or compute from .txt chain"""
    # Find files
    if not os.path.exists(chain_dir):
        print(f"Error: Directory {chain_dir} not found.")
        sys.exit(1)
        
    files = os.listdir(chain_dir)
    stats_file = None
    chain_file = None
    paramnames_file = None
    
    for f in files:
        if f.endswith('.stats'):
            stats_file = os.path.join(chain_dir, f)
        elif f.endswith('.txt') and not ('dead' in f or 'resume' in f):
            chain_file = os.path.join(chain_dir, f)
        elif f.endswith('.paramnames'):
            paramnames_file = os.path.join(chain_dir, f)
            
    if not paramnames_file:
        print(f"Error: .paramnames file not found in {chain_dir}")
        sys.exit(1)
        
    # Read param names
    with open(paramnames_file, 'r') as f:
        param_names = [line.strip().split()[0] for line in f if line.strip()]

    mean_params = {}
    
    # Try reading stats file
    if stats_file:
        print(f"Reading mean parameters from {stats_file}")
        with open(stats_file, 'r') as f:
            lines = f.readlines()
            start_idx = -1
            for i, line in enumerate(lines):
                if "Dim No.       Mean        Sigma" in line:
                    start_idx = i + 1
                    break
            
            if start_idx != -1:
                for line in lines[start_idx:]:
                    if "----------------" in line:
                        break
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        try:
                            idx = int(parts[0]) - 1 # 0-indexed
                            mean_val = float(parts[1])
                            if idx < len(param_names):
                                name = param_names[idx]
                                mean_params[name] = mean_val
                        except ValueError:
                            continue
    elif chain_file:
        print(f"Stats file not found, computing means from chain file {chain_file}...")
        data = np.loadtxt(chain_file)
        weights = data[:, 0]
        samples = data[:, 2:]
        for i, name in enumerate(param_names):
            mean_val = np.average(samples[:, i], weights=weights)
            mean_params[name] = mean_val
    else:
        print(f"Error: Neither .stats nor .txt chain file found in {chain_dir}")
        sys.exit(1)
        
    return mean_params

# Mock classes to bypass data loading
def mock_get_forest_weights(self, main_tracer):
    # Mock weights centered at z=2.33
    loglam_min = np.log10(3600)
    loglam_max = np.log10(4600)
    dloglam = 1e-4
    loglam = np.arange(loglam_min, loglam_max, dloglam)
    wave = 10**loglam
    weights = np.ones_like(wave)
    return wave, weights

def mock_get_qso_weights(self, tracer):
    z = np.linspace(2.1, 2.56, 1000)
    weights = np.ones_like(z)
    return z, weights

class MockData:
    def __init__(self, corr_item):
        self.corr_item = corr_item
        self.tracer1 = corr_item.tracer1
        self.tracer2 = corr_item.tracer2
        
        # Use FINE grid for smooth plotting
        # 500 bins -> 0.4 Mpc/h resolution
        rp_min, rp_max, rp_nbins = 0.0, 200.0, 50
        rt_max, rt_nbins = 200.0, 50
        
        # Default z_eff
        z_eff = 2.33 
        
        # Initialize cosmo params
        self.cosmo_params = {
            'Omega_m': 0.315, 'Omega_k': 0.0, 'Omega_r': 0.0, 'wl': -1.0
        }
        self.corr_item.cosmo_params = self.cosmo_params

        # Create coordinates
        self.model_coordinates = Coordinates(
            rp_min, rp_max, rt_max, rp_nbins, rt_nbins, z_eff=z_eff
        )
        self.data_coordinates = self.model_coordinates
        self.dist_model_coordinates = self.model_coordinates
        
        # Initialize corr_item coordinates
        self.corr_item.init_coordinates(
            self.model_coordinates, self.dist_model_coordinates, self.data_coordinates
        )
        
        # Attributes
        self.blind = False
        self.has_distortion = True 
        self.has_cov_mat = False 
        self.use_metal_autos = corr_item.config['model'].getboolean('use_metal_autos', True)
        
        # Dummy data vector (zeros)
        self.data_vec = np.zeros(self.model_coordinates.r_grid.size)
        self.cov_mat = None 
        
        # Dummy distortion matrix (Identity)
        # Use Gaussian smoothing along RP to mimic resolution effects
        # Typical resolution is ~3 Mpc/h
        # Bin size is 0.4 Mpc/h
        self.rp_sigma = 3.0 / 0.4 # pixels
        print(f"Applying Gaussian smoothing with sigma = {self.rp_sigma:.2f} pixels (3.0 Mpc/h)")
        self._distortion_mat = SmoothingMatrix(rp_nbins, rt_nbins, self.rp_sigma)
        
        # Cuts
        self.r_min_cut = 10.
        self.r_max_cut = 180.
        self.mu_min_cut = 0.
        self.mu_max_cut = 1.
        
        if 'broadband' in corr_item.config:
             self.coeff_binning_model = 1
             self.corr_item.init_broadband(self.coeff_binning_model)

        # Initialize Metals if configured
        if 'metals' in corr_item.config:
            self.metal_mats = {}
            self.metal_coordinates = {}
            if corr_item.new_metals:
                metals_in_tracer1, metals_in_tracer2, tracer_catalog = self._init_metal_tracers(
                    corr_item.config['metals'])
                metal_correlations = self._init_metal_correlations(
                    corr_item.config['metals'], metals_in_tracer1, metals_in_tracer2)
                self.corr_item.init_metals(tracer_catalog, metal_correlations)

    @property
    def distortion_mat(self):
        return self._distortion_mat

    def _init_metal_tracers(self, metal_config):
        metals_in_tracer1 = metal_config.get('in tracer1', '').split() if 'in tracer1' in metal_config else None
        metals_in_tracer2 = metal_config.get('in tracer2', '').split() if 'in tracer2' in metal_config else None
        
        tracer_catalog = {}
        
        # Add main tracers
        tracer_catalog[self.tracer1['name']] = {
            'name': self.tracer1['name'], 
            'type': self.tracer1['type'],
            'z': np.array([2.33]), 
            'weights': np.array([1.0])
        }
        if self.tracer2['name'] != self.tracer1['name']:
            tracer_catalog[self.tracer2['name']] = {
                'name': self.tracer2['name'], 
                'type': self.tracer2['type'],
                'z': np.array([2.33]), 
                'weights': np.array([1.0])
            }
            
        all_metals = set()
        if metals_in_tracer1: all_metals.update(metals_in_tracer1)
        if metals_in_tracer2: all_metals.update(metals_in_tracer2)
        
        for m in all_metals:
            tracer_catalog[m] = {
                'name': m, 
                'type': 'continuous',
                'z': np.array([2.33]), 
                'weights': np.array([1.0])
            } 
            
        return metals_in_tracer1, metals_in_tracer2, tracer_catalog

    @property
    def full_data_size(self):
        return self.model_coordinates.r_grid.size

    def _use_correlation(self, name1, name2):
        if name1 == 'CIV(eff)' or name2 == 'CIV(eff)':
            return name1 == name2
        if 'SiII' in name1 and 'SiII' in name2 and not self.use_metal_autos:
            return False
        else:
            return True

    def _init_metal_correlations(self, metal_config, metals_in_tracer1, metals_in_tracer2):
        metal_correlations = []
        if 'in tracer2' in metal_config:
            for metal in metals_in_tracer2:
                if not self._use_correlation(self.tracer1['name'], metal):
                    continue
                metal_correlations.append((self.tracer1['name'], metal))
        
        if 'in tracer1' in metal_config:
            for metal in metals_in_tracer1:
                if not self._use_correlation(metal, self.tracer2['name']):
                    continue
                metal_correlations.append((metal, self.tracer2['name']))
                
        if ('in tracer1' in metal_config) and ('in tracer2' in metal_config):
            for i, metal1 in enumerate(metals_in_tracer1):
                j0 = i if self.tracer1 == self.tracer2 else 0
                for metal2 in metals_in_tracer2[j0:]:
                    if not self._use_correlation(metal1, metal2):
                        continue
                    metal_correlations.append((metal1, metal2))
        
        for tracers in metal_correlations:
            self.metal_mats[tracers] = identity(self.full_data_size, format='csr')
            self.metal_coordinates[tracers] = Coordinates.init_from_grids(
                self.model_coordinates,
                self.model_coordinates.rp_grid,
                self.model_coordinates.rt_grid,
                self.model_coordinates.z_grid
            )
            
        return metal_correlations

def interpolate_to_finer_grid(m_input, n_rp, n_rt, rp_min, rp_max, rt_min, rt_max, scale_factor=4):
    """
    Interpolate the input 2D data to a finer grid using 2D spline interpolation.
    """
    # Original grid points (midpoints)
    drp = (rp_max - rp_min) / n_rp
    drt = (rt_max - rt_min) / n_rt
    rp_centers = np.linspace(rp_min + drp/2, rp_max - drp/2, n_rp)
    rt_centers = np.linspace(rt_min + drt/2, rt_max - drt/2, n_rt)
    
    # Reshape input to 2D grid
    try:
        grid_data = m_input.reshape((n_rp, n_rt))
    except ValueError:
        print("Warning: reshape to (n_rp,n_rt) failed; skipping interpolation.")
        return m_input, n_rp, n_rt

    # Create interpolator
    # Use linear interpolation (kx=1, ky=1) to avoid overshooting
    interp_func = RectBivariateSpline(rp_centers, rt_centers, grid_data, kx=1, ky=1)
    
    # New finer grid
    n_rp_new = n_rp * scale_factor
    n_rt_new = n_rt * scale_factor
    
    drp_new = (rp_max - rp_min) / n_rp_new
    drt_new = (rt_max - rt_min) / n_rt_new
    
    rp_centers_new = np.linspace(rp_min + drp_new/2, rp_max - drp_new/2, n_rp_new)
    rt_centers_new = np.linspace(rt_min + drt_new/2, rt_max - drt_new/2, n_rt_new)
    
    new_grid_data = interp_func(rp_centers_new, rt_centers_new)
    
    # Flatten back to 1D
    m_input_new = new_grid_data.flatten()
    
    return m_input_new, n_rp_new, n_rt_new

def get_smooth_ratio(y1, y2, x):
    sigma = 1.5
    num = y2 * y1
    den = y1 * y1
    num_smooth = gaussian_filter1d(num, sigma=sigma)
    den_smooth = gaussian_filter1d(den, sigma=sigma)
    ratio_smooth = num_smooth / (den_smooth + 1e-30)
    return ratio_smooth

def main():
    args = parse_args()
    
    # Determine mode
    mode = 'wedge'
    if args.compare:
        mode = 'compare'
    
    # Resolve configs
    config1 = args.config1 if args.config1 else args.config
    config2 = args.config2 if args.config2 else (args.config1 if args.config1 else args.config)

    # Initialize Vega for chain1
    print("Initializing Vega for chain1...")
    if not os.path.exists(config1):
        print(f"Error: Config file {config1} not found")
        sys.exit(1)
    try:
        vega_interface1 = VegaInterface(config1)
        using_mock_data1 = False
        print("Successfully initialized Vega with real data (chain1).")
    except Exception as e:
        print(f"Warning: Failed to initialize Vega (chain1) with real data: {e}")
        print("Falling back to MockData for chain1. NOTE: Plots will lack resolution effects (distortion) and may appear sharper/deeper than fitted results.")
        vega.metals.Metals.get_forest_weights = mock_get_forest_weights
        vega.metals.Metals.get_qso_weights = mock_get_qso_weights
        vega.data.Data = MockData
        vega_interface1 = VegaInterface(config1)
        using_mock_data1 = True

    # Initialize Vega for chain2 (if needed)
    vega_interface2 = None
    using_mock_data2 = None
    if args.chain2:
        print("Initializing Vega for chain2...")
        if not os.path.exists(config2):
            print(f"Error: Config file {config2} not found")
            sys.exit(1)
        try:
            vega_interface2 = VegaInterface(config2)
            using_mock_data2 = False
            print("Successfully initialized Vega with real data (chain2).")
        except Exception as e:
            print(f"Warning: Failed to initialize Vega (chain2) with real data: {e}")
            print("Falling back to MockData for chain2. NOTE: Plots will lack resolution effects (distortion) and may appear sharper/deeper than fitted results.")
            vega.metals.Metals.get_forest_weights = mock_get_forest_weights
            vega.metals.Metals.get_qso_weights = mock_get_qso_weights
            vega.data.Data = MockData
            vega_interface2 = VegaInterface(config2)
            using_mock_data2 = True

    # Check if correlation exists
    if args.corr not in vega_interface1.data:
        # If using MockData, data dict might be empty or initialized differently?
        # VegaInterface init calls self.data[name] = data.Data(corr_item) which is MockData
        # So it should be fine.
        if args.corr not in vega_interface1.data:
             print(f"Error: Correlation {args.corr} not found in config1. Available: {list(vega_interface1.data.keys())}")
             sys.exit(1)
    if vega_interface2 is not None and args.corr not in vega_interface2.data:
        print(f"Error: Correlation {args.corr} not found in config2. Available: {list(vega_interface2.data.keys())}")
        sys.exit(1)

    # Get Data coordinates info
    data_obj = vega_interface1.data[args.corr]
    rp_min = data_obj.data_coordinates.rp_min
    rp_max = data_obj.data_coordinates.rp_max
    rt_max = data_obj.data_coordinates.rt_max
    n_rp = data_obj.data_coordinates.rp_nbins
    n_rt = data_obj.data_coordinates.rt_nbins
    rt_min = 0.0
    if vega_interface2 is not None:
        data_obj2 = vega_interface2.data[args.corr]
        rp_min2 = data_obj2.data_coordinates.rp_min
        rp_max2 = data_obj2.data_coordinates.rp_max
        rt_max2 = data_obj2.data_coordinates.rt_max
        n_rp2 = data_obj2.data_coordinates.rp_nbins
        n_rt2 = data_obj2.data_coordinates.rt_nbins

    # Read params for chain 1
    params1 = read_params(args.chain1)
    
    # Read params for chain 2 if needed
    params2 = None
    if args.chain2:
        params2 = read_params(args.chain2)
    elif mode == 'compare':
        print("Warning: Compare mode requested but --chain2 not provided. Comparing chain1 with itself (ratio=1).")
        params2 = params1

    # Update params and compute model 1
    print("Computing model for chain 1...")
    # Update only params that exist in Vega config
    current_params = vega_interface1.params.copy()
    for k, v in params1.items():
        if k in current_params:
            current_params[k] = v
    
    models1 = vega_interface1.compute_model(current_params, run_init=True)
    if args.corr not in models1:
        print(f"Error: Correlation {args.corr} not found in model output. Available: {list(models1.keys())}")
        sys.exit(1)
    
    # Interpolate to finer grid
    scale_factor = 10
    m1_coarse = models1[args.corr]
    
    if using_mock_data1:
        m1_fine = m1_coarse
        n_rp_fine = n_rp
        n_rt_fine = n_rt
        rp_centers_active = np.linspace(rp_min + (rp_max - rp_min)/(2*n_rp_fine), rp_max - (rp_max - rp_min)/(2*n_rp_fine), n_rp_fine)
        rt_centers_active = np.linspace(rt_min + (rt_max - rt_min)/(2*n_rt_fine), rt_max - (rt_max - rt_min)/(2*n_rt_fine), n_rt_fine)
    else:
        def _choose_grids_for_vec(data_obj_local, vec_len):
            cands = []
            try:
                cands.append((data_obj_local.model_coordinates.rp_regular_grid, data_obj_local.model_coordinates.rt_regular_grid))
            except Exception:
                pass
            try:
                cands.append((data_obj_local.dist_model_coordinates.rp_regular_grid, data_obj_local.dist_model_coordinates.rt_regular_grid))
            except Exception:
                pass
            try:
                cands.append((data_obj_local.data_coordinates.rp_regular_grid, data_obj_local.data_coordinates.rt_regular_grid))
            except Exception:
                pass
            for rp_g, rt_g in cands:
                if rp_g is not None and rt_g is not None and rp_g.size == vec_len and rt_g.size == vec_len:
                    return rp_g, rt_g
            return None, None
        rp_g0, rt_g0 = _choose_grids_for_vec(data_obj, m1_coarse.size)
        if rp_g0 is not None and rt_g0 is not None:
            rp_centers0 = np.unique(rp_g0)
            rt_centers0 = np.unique(rt_g0)
            scale_factor = 10
            try:
                grid0 = m1_coarse.reshape((rp_centers0.size, rt_centers0.size))
                interp_func = RectBivariateSpline(rp_centers0, rt_centers0, grid0, kx=1, ky=1)
                n_rp_fine = rp_centers0.size * scale_factor
                n_rt_fine = rt_centers0.size * scale_factor
                drp0 = rp_centers0[1] - rp_centers0[0] if rp_centers0.size > 1 else (rp_max - rp_min)
                drt0 = rt_centers0[1] - rt_centers0[0] if rt_centers0.size > 1 else (rt_max - rt_min)
                rp_centers_active = np.linspace(rp_centers0[0] - drp0/2 + drp0/(2*scale_factor), rp_centers0[-1] + drp0/2 - drp0/(2*scale_factor), n_rp_fine)
                rt_centers_active = np.linspace(rt_centers0[0] - drt0/2 + drt0/(2*scale_factor), rt_centers0[-1] + drt0/2 - drt0/(2*scale_factor), n_rt_fine)
                new_grid = interp_func(rp_centers_active, rt_centers_active)
                m1_fine = new_grid.flatten()
            except Exception:
                m1_fine = m1_coarse
                n_rp_fine = rp_centers0.size
                n_rt_fine = rt_centers0.size
                rp_centers_active = rp_centers0
                rt_centers_active = rt_centers0
        else:
            scale_factor = 10
            m1_fine, n_rp_fine, n_rt_fine = interpolate_to_finer_grid(
                m1_coarse, n_rp, n_rt, rp_min, rp_max, rt_min, rt_max, scale_factor
            )
            rp_centers_active = np.linspace(rp_min + (rp_max - rp_min)/(2*n_rp_fine), rp_max - (rp_max - rp_min)/(2*n_rp_fine), n_rp_fine)
            rt_centers_active = np.linspace(rt_min + (rt_max - rt_min)/(2*n_rt_fine), rt_max - (rt_max - rt_min)/(2*n_rt_fine), n_rt_fine)

    L1 = m1_fine.size
    if L1 != n_rp_fine * n_rt_fine:
        print(f"Warning: model length {L1} != n_rp*n_rt ({n_rp_fine*n_rt_fine}); reconciling grid.")
        if L1 % n_rt_fine == 0:
            n_rp_fine = L1 // n_rt_fine
        elif L1 % n_rp_fine == 0:
            n_rt_fine = L1 // n_rp_fine
    
    # Compute model 2 if needed
    m2_fine = None
    if params2:
        print("Computing model for chain 2...")
        # Choose interface for chain2
        vega_if2 = vega_interface2 if vega_interface2 is not None else vega_interface1
        current_params = vega_if2.params.copy()
        for k, v in params2.items():
            if k in current_params:
                current_params[k] = v
        models2 = vega_if2.compute_model(current_params, run_init=True)
        m2_coarse = models2[args.corr]
        
        if (using_mock_data2 if using_mock_data2 is not None else using_mock_data1):
            m2_fine = m2_coarse
            rp_centers_active2 = rp_centers_active
            rt_centers_active2 = rt_centers_active
        else:
            data_obj_local = data_obj2 if vega_interface2 is not None else data_obj
            rp_g02, rt_g02 = _choose_grids_for_vec(data_obj_local, m2_coarse.size)
            if rp_g02 is not None and rt_g02 is not None:
                rp_centers02 = np.unique(rp_g02)
                rt_centers02 = np.unique(rt_g02)
                try:
                    grid02 = m2_coarse.reshape((rp_centers02.size, rt_centers02.size))
                    interp_func2 = RectBivariateSpline(rp_centers02, rt_centers02, grid02, kx=1, ky=1)
                    n_rp_fine2 = rp_centers02.size * scale_factor
                    n_rt_fine2 = rt_centers02.size * scale_factor
                    drp02 = rp_centers02[1] - rp_centers02[0] if rp_centers02.size > 1 else ((rp_max2 if vega_interface2 is not None else rp_max) - (rp_min2 if vega_interface2 is not None else rp_min))
                    drt02 = rt_centers02[1] - rt_centers02[0] if rt_centers02.size > 1 else ((rt_max2 if vega_interface2 is not None else rt_max) - rt_min)
                    rp_centers_active2 = np.linspace(rp_centers02[0] - drp02/2 + drp02/(2*scale_factor), rp_centers02[-1] + drp02/2 - drp02/(2*scale_factor), n_rp_fine2)
                    rt_centers_active2 = np.linspace(rt_centers02[0] - drt02/2 + drt02/(2*scale_factor), rt_centers02[-1] + drt02/2 - drt02/(2*scale_factor), n_rt_fine2)
                    new_grid2 = interp_func2(rp_centers_active2, rt_centers_active2)
                    m2_fine = new_grid2.flatten()
                except Exception:
                    m2_fine = m2_coarse
                    rp_centers_active2 = rp_centers02
                    rt_centers_active2 = rt_centers02
            else:
                m2_fine = m2_coarse
                rp_centers_active2 = rp_centers_active
                rt_centers_active2 = rt_centers_active

        # Reconcile grid for model 2 as well
        if m2_fine is not None:
            L2 = m2_fine.size
            if L2 != n_rp_fine * n_rt_fine:
                print(f"Warning: model2 length {L2} != n_rp*n_rt ({n_rp_fine*n_rt_fine}); reconciling grid.")
                if L2 % n_rt_fine == 0:
                    n_rp_fine = L2 // n_rt_fine
                    print(f"Adjusted n_rp to {n_rp_fine} (n_rt stays {n_rt_fine}).")
                elif L2 % n_rp_fine == 0:
                    n_rt_fine = L2 // n_rp_fine
                    print(f"Adjusted n_rt to {n_rt_fine} (n_rp stays {n_rp_fine}).")
                else:
                    best_n_rp = None
                    for k in range(1, int(np.sqrt(L2)) + 1):
                        if L2 % k == 0:
                            if best_n_rp is None or abs(k - n_rp_fine) < abs(best_n_rp - n_rp_fine):
                                best_n_rp = k
                            if abs((L2 // k) - n_rp_fine) < abs(best_n_rp - n_rp_fine):
                                best_n_rp = L2 // k
                    if best_n_rp is not None:
                        n_rp_fine = best_n_rp
                        n_rt_fine = L2 // best_n_rp
                        print(f"Guessed grid2 to ({n_rp_fine}, {n_rt_fine}).")
                    else:
                        print("Warning: failed to factorize grid2 cleanly; proceeding may fail.")

    # Plotting setup
    plt.rcParams.update({'font.size': 14})
    plt.rcParams.update({'axes.titlesize': 18})
    plt.rcParams.update({'axes.labelsize': 16})
    plt.rcParams.update({'xtick.labelsize': 14})
    plt.rcParams.update({'ytick.labelsize': 14})
    plt.rcParams.update({'legend.fontsize': 16})
    plt.rcParams.update({'lines.linewidth': 2.5})
    
    mu_bins = [(0.95, 1.0), (0.8, 0.95), (0.5, 0.8), (0.0, 0.5)]
    
    # Output grid (Coarse, for Wedge r bins)
    n_rp_out = 50
    r_min_plot = args.plot_rmin
    r_max_plot = args.plot_rmax

    rt_mesh, rp_mesh = np.meshgrid(rt_centers_active, rp_centers_active)
    vol_weights = rt_mesh.flatten()

    # Direct wedge calculator (avoids reliance on (n_rp,n_rt) when using real data)
    def compute_wedge_direct(data_vec, rp_grid, rt_grid, mu_range, r_min, r_max, n_r, abs_mu=True):
        r = np.sqrt(rp_grid**2 + rt_grid**2)
        mu = rp_grid / np.maximum(r, 1e-12)
        if abs_mu:
            mu = np.abs(mu)
        mu_min, mu_max = mu_range
        r_edges = np.linspace(r_min, r_max, n_r + 1)
        r_centers = (r_edges[:-1] + r_edges[1:]) / 2.0
        mask_mu = (mu >= mu_min) & (mu <= mu_max)
        # Volume weights ~ rt
        w = rt_grid
        wedge_vals = np.zeros(n_r)
        for i in range(n_r):
            in_bin = mask_mu & (r >= r_edges[i]) & (r < r_edges[i+1])
            if np.any(in_bin):
                wi = w[in_bin]
                di = data_vec[in_bin]
                s = wi.sum()
                wedge_vals[i] = (wi * di).sum() / s if s > 0 else 0.0
            else:
                wedge_vals[i] = 0.0
        return r_centers, wedge_vals

    def compute_wedge_with_errors(data_obj_local, mu_range, n_r_out, abs_mu=True):
        rp_min = data_obj_local.data_coordinates.rp_min
        rp_max = data_obj_local.data_coordinates.rp_max
        rt_max = data_obj_local.data_coordinates.rt_max
        n_rp = data_obj_local.data_coordinates.rp_nbins
        n_rt = data_obj_local.data_coordinates.rt_nbins
        w_obj = Wedge(mu=mu_range,
                      rp=(rp_min, rp_max, n_rp),
                      rt=(0.0, rt_max, n_rt),
                      r=(0.0, 200.0, 50),
                      abs_mu=abs_mu)
        data_vec = data_obj_local.data_vec
        try:
            cov = data_obj_local.cov_mat
            rd, dd, cd = w_obj(data_vec, covariance=cov)
            errs = np.sqrt(np.diag(cd))
        except Exception:
            rd, dd = w_obj(data_vec)
            errs = np.zeros_like(dd)
        return rd, dd, errs

    # Helper: choose coordinate grids matching a given model vector length
    def choose_grids_for_model(data_obj, vec_len):
        candidates = []
        try:
            candidates.append(('model_grid', data_obj.model_coordinates.rp_grid, data_obj.model_coordinates.rt_grid))
        except Exception:
            pass
        try:
            candidates.append(('model_regular', data_obj.model_coordinates.rp_regular_grid, data_obj.model_coordinates.rt_regular_grid))
        except Exception:
            pass
        try:
            candidates.append(('dist_model_regular', data_obj.dist_model_coordinates.rp_regular_grid, data_obj.dist_model_coordinates.rt_regular_grid))
        except Exception:
            pass
        try:
            candidates.append(('data_regular', data_obj.data_coordinates.rp_regular_grid, data_obj.data_coordinates.rt_regular_grid))
        except Exception:
            pass
        for name, rp_g, rt_g in candidates:
            if rp_g is not None and rt_g is not None and rp_g.size == vec_len and rt_g.size == vec_len:
                print(f"Using {name} grids for wedge (matched length {vec_len}).")
                return rp_g, rt_g
        # Fallback: if any candidate has matching product but not flattened length
        for name, rp_g, rt_g in candidates:
            if rp_g is not None and rt_g is not None:
                if rp_g.size == vec_len:
                    print(f"Using {name} rp_grid and guessing rt_grid to same length; results may be unreliable.")
                    return rp_g, rt_g
        raise RuntimeError(f"No coordinate grids match model length {vec_len}.")

    # Setup figure
    if mode == 'compare' and args.one:
        fig, ax = plt.subplots(figsize=(10, 8))
        axes = [ax] * 4
    else:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
    colors = ['C0', 'C1', 'C2', 'C3']
    linestyles = ['-', '--', '-.', ':']

    # Helper for auto y-limits in compare-one mode
    def _round_up_strict(x, step=0.05):
        k = np.floor(x / step) + 1.0
        return np.round(k * step, 2)
    def _round_down_strict(x, step=0.05):
        k = np.ceil(x / step) - 1.0
        return np.round(k * step, 2)

    ratio_curves_one = [] if (mode == 'compare' and args.one) else None

    for i, (mu_min, mu_max) in enumerate(mu_bins):
        ax = axes[i]
        
        if using_mock_data1:
            w = Wedge(mu=(mu_min, mu_max),
                      rp=(rp_min, rp_max, n_rp_fine),
                      rt=(rt_min, rt_max, n_rt_fine),
                      r=(r_min_plot, r_max_plot, n_rp_out),
                      abs_mu=True)
            r, wedge1 = w(m1_fine, weights=vol_weights)
            wedge2 = None
            if m2_fine is not None:
                r, wedge2 = w(m2_fine, weights=vol_weights)
        else:
            rp_centers = rp_centers_active
            rt_centers = rt_centers_active
            n_rp_vec = rp_centers.size
            n_rt_vec = rt_centers.size
            if n_rp_vec > 1:
                drp_vec = rp_centers[1] - rp_centers[0]
            else:
                drp_vec = (rp_max - rp_min) / max(1, n_rp_out)
            if n_rt_vec > 1:
                drt_vec = rt_centers[1] - rt_centers[0]
            else:
                drt_vec = (rt_max - rt_min) / max(1, n_rp_out)
            rp_min_edge = rp_centers[0] - drp_vec/2
            rp_max_edge = rp_centers[-1] + drp_vec/2
            rt_min_edge = max(0.0, rt_centers[0] - drt_vec/2)
            rt_max_edge = rt_centers[-1] + drt_vec/2
            rt_mesh_vec, rp_mesh_vec = np.meshgrid(rt_centers, rp_centers)
            vol_weights_vec = rt_mesh_vec.flatten()
            w = Wedge(mu=(mu_min, mu_max),
                      rp=(rp_min_edge, rp_max_edge, n_rp_vec),
                      rt=(rt_min_edge, rt_max_edge, n_rt_vec),
                      r=(r_min_plot, r_max_plot, n_rp_out),
                      abs_mu=True)
            r, wedge1 = w(m1_fine, weights=vol_weights_vec)
            wedge2 = None
            if m2_fine is not None:
                rp_centers2 = rp_centers_active2 if 'rp_centers_active2' in locals() else rp_centers
                rt_centers2 = rt_centers_active2 if 'rt_centers_active2' in locals() else rt_centers
                if rp_centers2.size != n_rp_vec or rt_centers2.size != n_rt_vec or np.any(np.abs(rp_centers2 - rp_centers) > 1e-8) or np.any(np.abs(rt_centers2 - rt_centers) > 1e-8):
                    rt_mesh_vec2, rp_mesh_vec2 = np.meshgrid(rt_centers2, rp_centers2)
                    vol_weights_vec2 = rt_mesh_vec2.flatten()
                    if rp_centers2.size > 1:
                        drp_vec2 = rp_centers2[1] - rp_centers2[0]
                    else:
                        drp_vec2 = drp_vec
                    if rt_centers2.size > 1:
                        drt_vec2 = rt_centers2[1] - rt_centers2[0]
                    else:
                        drt_vec2 = drt_vec
                    rp_min_edge2 = rp_centers2[0] - drp_vec2/2
                    rp_max_edge2 = rp_centers2[-1] + drp_vec2/2
                    rt_min_edge2 = max(0.0, rt_centers2[0] - drt_vec2/2)
                    rt_max_edge2 = rt_centers2[-1] + drt_vec2/2
                    w2 = Wedge(mu=(mu_min, mu_max),
                               rp=(rp_min_edge2, rp_max_edge2, rp_centers2.size),
                               rt=(rt_min_edge2, rt_max_edge2, rt_centers2.size),
                               r=(r_min_plot, r_max_plot, n_rp_out),
                               abs_mu=True)
                    _, wedge2 = w2(m2_fine, weights=vol_weights_vec2)
                else:
                    _, wedge2 = w(m2_fine, weights=vol_weights_vec)
            
        if mode == 'wedge':
            ax.set_xlim(0, 200)
            ax.grid(linestyle=':', alpha=0.5)
            smooth_sigma = 0.5
            y1 = gaussian_filter1d(wedge1, sigma=smooth_sigma) * r**2
            ax.plot(r, y1, '-', label='Without Reionization')
            if not using_mock_data1:
                r_d, w_d, e_d = compute_wedge_with_errors(data_obj, (mu_min, mu_max), n_rp_out, True)
                ax.errorbar(r_d, w_d * r_d**2, yerr=e_d * r_d**2, fmt='o', color='k', markersize=6, alpha=0.8, label='DESI DR1')
            if wedge2 is not None:
                y2 = gaussian_filter1d(wedge2, sigma=smooth_sigma) * r**2
                ax.plot(r, y2, ':', label='With Reionization')
                y_min = min(np.min(y1), np.min(y2))
                y_max = max(np.max(y1), np.max(y2))
            else:
                y_min = np.min(y1)
                y_max = np.max(y1)
            pad = 0.05 * max(1e-12, (y_max - y_min))
            ax.set_ylim(y_min - pad, y_max + pad)
                
            if i % 2 == 0:
                ax.set_ylabel(r'$r^2 \xi(r)$')
            if i == 0:
                ax.legend(frameon=False)
            ax.set_title(r'${} < \mu < {}$'.format(mu_min, mu_max))
            if i >= 2:
                ax.set_xlabel('r [Mpc/h]')
                
        elif mode == 'compare':
            ax.set_xlim(10, 180)
            
            ratio = get_smooth_ratio(wedge1, wedge2, r)
            
            if not args.one:
                ax.plot(r, ratio, 'k-')
                if i % 2 == 0:
                    ax.set_ylabel('Ratio')
                ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
                if i == 0:
                    ax.legend(frameon=False)
                ax.set_title(r'${} < \mu < {}$'.format(mu_min, mu_max))
                if i >= 2:
                    ax.set_xlabel('r [Mpc/h]')
            else:
                label_text = r'${} < \mu < {}$'.format(mu_min, mu_max)
                ax.plot(r, ratio, color=colors[i], linestyle=linestyles[i], label=label_text)
                ratio_curves_one.append(ratio)
                if i == 3:
                    ax.set_ylabel('Ratio')
                    ax.set_xlabel('r [Mpc/h]')
                    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
                    # Auto y-limits across all lines with 0.05 grid (strict bounds)
                    try:
                        rmin = np.nanmin([np.nanmin(rc) for rc in ratio_curves_one])
                        rmax = np.nanmax([np.nanmax(rc) for rc in ratio_curves_one])
                        ylo = _round_down_strict(rmin, 0.05)
                        yhi = _round_up_strict(rmax, 0.05)
                        ax.set_ylim(ylo, yhi)
                    except Exception:
                        pass
                    ax.legend(frameon=False)

    if not (mode == 'compare' and args.one):
        plt.subplots_adjust(hspace=0.13, wspace=0.1)
    out_dir = os.path.dirname(__file__)
    chain1_name = os.path.basename(os.path.normpath(args.chain1)) if args.chain1 else 'chain1'
    chain2_name = os.path.basename(os.path.normpath(args.chain2)) if args.chain2 else chain1_name
    if mode == 'compare':
        style = 'one' if args.one else 'four'
        fname = f"polychord_{args.corr}_compare_{chain1_name}_vs_{chain2_name}_{style}.png"
    else:
        if args.chain2:
            fname = f"polychord_{args.corr}_wedge_{chain1_name}_vs_{chain2_name}.png"
        else:
            fname = f"polychord_{args.corr}_wedge_{chain1_name}.png"
    out_path = os.path.join(out_dir, fname)
    try:
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        print(f"Saved figure to {out_path}")
    except Exception as e:
        print(f"Failed to save figure: {e}")
        
    plt.show()

if __name__ == "__main__":
    main()
