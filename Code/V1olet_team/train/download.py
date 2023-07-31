import gdown
import os

des = 'download'

try:
    os.mkdir(des)
except:
    pass

# url = "https://drive.google.com/file/d/116RCeKh-GdNKkcG5Bh4Lj0hQjaAofHvA/view?usp=share_link"
# output =  f"{des}/shrec23_test_final_dataset.zip"
# gdown.download(url=url, output=output, quiet=False, fuzzy=True)

# url = "https://drive.google.com/file/d/1vJuJAw2GdRAaHNboJglitqTKrwl0HVhK/view?usp=share_link"
# output =  f"{des}/shrec23_test_predict.zip"
# gdown.download(url=url, output=output, quiet=False, fuzzy=True)

# url = "https://drive.google.com/file/d/1kEa1NFX0_FkrQM1aLwgIDpAt5sqRRs8I/view?usp=share_link"
# output =  f"{des}/shrec23_train_merge_final_dataset.zip"
# gdown.download(url=url, output=output, quiet=False, fuzzy=True)

url = "https://drive.google.com/file/d/1ZB3dcGpI9OhMzpbEZlnYm9klaDiBVTxD/view?usp=drive_link"
output =  f"{des}/shrec23_test_final_dataset.zip"
gdown.download(url=url, output=output, quiet=False, fuzzy=True)

url = "https://drive.google.com/file/d/1OVgk5CaZeLIy6sgU_2G_MTOYgYmIgPws/view?usp=drive_link"
output =  f"{des}/shrec23_test_predict.zip"
gdown.download(url=url, output=output, quiet=False, fuzzy=True)

url = "https://drive.google.com/file/d/1Q2mXtPj8VxFCZXWbdWTfmOZ3H0cVQ35j/view?usp=drive_link"
output =  f"{des}/shrec23_train_merge_final_dataset.zip"
gdown.download(url=url, output=output, quiet=False, fuzzy=True)

