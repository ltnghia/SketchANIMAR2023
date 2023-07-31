import gdown
import os

des = 'download'

try:
    os.mkdir(des)
except:
    pass

url = "https://drive.google.com/file/d/1YMXwb3NpA_1kdXL3i--m7QJihq7Qp5sZ/view?usp=share_link"
output =  f"{des}/SketchQuery_Test.zip"
gdown.download(url=url, output=output, quiet=False, fuzzy=True)

url = "https://drive.google.com/file/d/1qzOKhHO1CqK9l382MfTGt9kWZaLwIli9/view?usp=share_link"
output =  f"{des}/0_3_NN_best_model_sketch_EfficientNetV2S_384_711.h5"
gdown.download(url=url, output=output, quiet=False, fuzzy=True)

url = "https://drive.google.com/file/d/18TtpH2-LtVhA0Ws-YqAd69aftUAaYE_8/view?usp=share_link"
output =  f"{des}/0_4_NN_best_model_sketch_ConvNeXtTiny_384_711.h5"
gdown.download(url=url, output=output, quiet=False, fuzzy=True)

url = "https://drive.google.com/file/d/1jKKosLNgtRtAi-K7BG8o6LahqkogpuYG/view?usp=share_link"
output =  f"{des}/0_23_NN_best_model_sketch_CAFormerS18_384_711.h5"
gdown.download(url=url, output=output, quiet=False, fuzzy=True)

url = "https://drive.google.com/file/d/1SjqG_bfeRDPXB-n1Gd6PLVIvmYAsXNlj/view?usp=share_link"
output =  f"{des}/0_26_NN_best_model_sketch_EfficientNetV1B5_384_711.h5"
gdown.download(url=url, output=output, quiet=False, fuzzy=True)

url = "https://drive.google.com/file/d/1vRoqkz-hQZCgFJAa--GQIihLpE4lFdPK/view?usp=share_link"
output =  f"{des}/last_hit_best_model_sketch_ConvNeXtSmall_384_711.h5"
gdown.download(url=url, output=output, quiet=False, fuzzy=True)

