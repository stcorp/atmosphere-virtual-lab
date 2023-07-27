import errno
import os

import boto3
import requests

# version of os.makedirs that won't complain if the path already exists
def make_path(path):
    try:
        os.makedirs(path)
    except EnvironmentError as _error:
        if _error.errno != errno.EEXIST or not os.path.isdir(path):
            raise


def get_s3path(product):
    url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=Name eq '{product}'"
    result = requests.get(url, timeout=60, allow_redirects=False).json()
    if len(result['value']) == 0:
        raise Exception(f"could not find product '{product}' in CDSE catalogue")
    return result['value'][0]['S3Path']


def download(product, target_directory):
    endpoint_url = "https://eodata.dataspace.copernicus.eu/"
    bucket = "eodata"
    if "CDSE_S3_ACCESS" not in os.environ or "CDSE_S3_SECRET" not in os.environ:
        raise Exception("CDSE_S3_ACCESS and CDSE_S3_SECRET environment variables need to be set to download from CDSE. "
                        "CDSE S3 credentials can be obtained by following the instructions at "
                        "https://documentation.dataspace.copernicus.eu/APIs/S3.html")
    access_key = os.environ["CDSE_S3_ACCESS"]
    secret_key = os.environ["CDSE_S3_SECRET"]

    s3path = get_s3path(product).removeprefix(f"/{bucket}/")
    if os.path.basename(s3path) != product:
        s3path = os.path.join(s3path, product)

    resource = boto3.resource(service_name="s3", aws_access_key_id=access_key, aws_secret_access_key=secret_key,
                              endpoint_url=endpoint_url)
    objs = list(resource.Bucket(bucket).objects.filter(Prefix=s3path))
    if not objs:
        raise Exception(f"could not find product '{product}' in CDSE object store")

    basepath = os.path.dirname(s3path)
    for obj in objs:
        target = os.path.join(target_directory, os.path.relpath(obj.key, basepath))
        if obj.key.endswith('/'):
            make_path(target)
        else:
            dirname = os.path.dirname(target)
            if dirname != '':
                make_path(dirname)
            resource.Object(bucket, obj.key).download_file(target)
