import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from vega.vega_interface import VegaInterface
vega = VegaInterface('examples/main.ini')
vega.compute_model()