# from anomalib.metrics.aupro import _AUPRO as AUPRO
# import torch
# #Create sample data
# labels = torch.randint(0, 2, (1, 10, 5))
# scores = torch.randint(0, 100, (1, 10, 5)).float() / 100.0
#
#
# fpr_limit = 0.3
# aupro = AUPRO(fpr_limit=fpr_limit)
# aupro.update(scores, labels)
# computed_aupro = aupro.compute()
# print(f"Computed AUPRO: {computed_aupro}")
#
#
from dataclasses import dataclass
import torch
from anomalib.metrics import AUROC

@dataclass
class Batch:
    preds: torch.Tensor
    target: torch.Tensor

labels = torch.randint(0, 2, (10,)).long()
scores = torch.rand_like(labels, dtype=torch.float)

batch = Batch(preds=scores, target=labels)
metric = AUROC(fields=["preds", "target"])
auroc_score = metric(batch)
print(auroc_score)




from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore
from anomalib.metrics import AUROC, F1Score, AUPRO
from anomalib.metrics.evaluator import Evaluator


image_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="image_")
image_f1score = F1Score(fields=["pred_label", "gt_label"], prefix="image_")
pixel_auroc = AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_", strict=False)
pixel_aupro = AUPRO(fields=["anomaly_map", "gt_mask"], prefix="pixel_", strict=False)

datamodule = MVTecAD(num_workers=0, category="toothbrush")

test_metrics = [image_auroc, image_f1score, pixel_auroc, pixel_aupro]
evaluator = Evaluator(test_metrics=test_metrics)


model = Patchcore(
    evaluator=evaluator,
)

engine = Engine()

engine.fit(datamodule=datamodule, model=model)
engine.test(datamodule=datamodule, model=model)
