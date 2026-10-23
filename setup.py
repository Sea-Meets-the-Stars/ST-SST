# Standard imports
import glob, os
from setuptools import setup, find_packages


# Begin setup
setup_keywords = dict()
setup_keywords['name'] = 'st_sst'
setup_keywords['description'] = 'Running the Scattering Transform on SST data'
setup_keywords['author'] = 'J. Xavier Prochaska'
setup_keywords['author_email'] = 'jxp@ucsc.edu'
setup_keywords['license'] = 'BSD'
setup_keywords['url'] = 'https://github.com/Sea-Meets-the-Stars/ST-SST'
setup_keywords['version'] = '0.1.0'
# Use README.md as long_description.
setup_keywords['long_description'] = ''
if os.path.exists('README.md'):
    with open('README.md') as readme:
        setup_keywords['long_description'] = readme.read()
setup_keywords['provides'] = [setup_keywords['name']]
setup_keywords['python_requires'] = '>=3.12'
setup_keywords['install_requires'] = [
    'numpy', 'scipy', 'pandas', 'pyarrow', 'matplotlib', 'seaborn',
    'scikit-learn', 'tqdm', 'IPython', 'pytest',
    # Oceanography / gridded data I/O
    'xarray', 'h5netcdf', 'h5py', 'netcdf4', 'cftime',
    # Cloud-hosted Zarr stores (LLC4320_v2 on Nautilus S3)
    'zarr', 's3fs', 'fsspec']
# The scattering-transform backend and sibling packages (e.g. wrangler) are
# not pinned here; install them via requirements.txt.
setup_keywords['zip_safe'] = False
setup_keywords['packages'] = find_packages()

if os.path.isdir('bin'):
    setup_keywords['scripts'] = [fname for fname in glob.glob(os.path.join('bin', '*'))
                                 if not os.path.basename(fname).endswith('.rst')]

setup(**setup_keywords)
