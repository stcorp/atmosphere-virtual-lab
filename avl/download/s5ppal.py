import os

from pystac_client import ItemSearch
import requests


def get_url(product):
    endpoint = "https://data-portal.s5p-pal.com/cat/sentinel-5p/search"
    items = list(ItemSearch(endpoint, ids=os.path.splitext(product)[0]).items())
    if len(items) == 0:
        raise Exception(f"could not find product '{product}' in S5P-PAL STAC catalogue")
    return items[0].assets["download"].href


def download(product, target_directory):
    url = get_url(product)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(os.path.join(target_directory, product), 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
