import zipfile
from zipfile import ZipFile
from concurrent.futures import ThreadPoolExecutor
import time
import os
from os.path import join as path_join

from_dir = 'download'
des = 'unzip'

try:
    os.mkdir(des)
except:
    pass

def do_unzip(zip_path, out_path):
    print(zip_path)
    try:
        start = time.time()
        with ZipFile(zip_path) as handle:
            handle.extractall(out_path)
    except:
        pass
    finally:
        print('Unzip', zip_path, 'Time:', time.time() - start)

filename = 'SketchQuery_Test.zip'
zip_path = path_join(from_dir, filename)
do_unzip(zip_path, des)

