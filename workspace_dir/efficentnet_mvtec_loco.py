
from anomalib.engine import Engine
from anomalib.models import EfficientAd
from anomalib.data import MVTecLOCO


from anomalib.models.image.efficient_ad.torch_model import (
    EfficientAdModel,
    EfficientAdModelSize
)
#
# datamodule = Folder(
#     name="pushpins",
#     root="/content/MvTec_Loco_AD/breakfast_box/",
#     normal_dir="train/good)",
#     train_batch_size=1,  # Number of images per training batch
#     eval_batch_size=32,  # Number of images per validation/test batch
#     num_workers=8,  # Number of parallel processes for data loading
# )
#
# datamodule.setup()
#
# model = EfficientAd(teacher_out_channels=384,
#     model_size=EfficientAdModelSize.M)

datamodule = MVTecLOCO(train_batch_size=1)
model = EfficientAd()

engine = Engine(max_epochs=200)

engine.fit(datamodule=datamodule, model=model)