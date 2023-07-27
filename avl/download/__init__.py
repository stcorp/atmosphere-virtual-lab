import os
import re

from . import avl
from . import cdse
from . import s5ppal

MAPPING = {
  r"^S1.*\.SAFE$": cdse.download,
  r"^S5P_PAL_.*\.nc$": s5ppal.download,
  r"^S5P.*\.nc$": cdse.download,
}

def download(products, target_directory="."):
    """
    Download product(s) from either CDSE, atmospherevirtuallab.org, skipping files
    that already exist.

    Arguments:
    products -- product file/directory name or list/tuple of file/directory names
    target_directory -- path where to store products (default '.')
    """
    if isinstance(products, (list, tuple)):
        for product in products:
            download(product)
        return

    product = os.path.basename(products)
    targetpath = os.path.join(target_directory, product)

    if os.path.exists(targetpath):
        return product

    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    for mapping in MAPPING:
        if re.match(mapping, product) is not None:
            download_backend = MAPPING[mapping]
            download_backend(product, target_directory)
            return

    # fileback to AVL archive
    avl.download(product, target_directory)
