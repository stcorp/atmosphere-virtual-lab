import os
import requests

def download(product, target_directory):
    url = "https://atmospherevirtuallab.org/files/" + product
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(os.path.join(target_directory, product), 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
