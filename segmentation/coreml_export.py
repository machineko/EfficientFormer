from time import time

import coremltools as ct
from backbone import efficientformer_l1_feat, efficientformer_l3_feat, efficientformer_l7_feat
import torch
from coreml_torch_utils import InputEnumeratedShapeImage, RenameOutput, OutputStaticImage

# l1 = efficientformer_l1_feat(pretrained="checkpoints/efficientformer_l1_1000d.pth")
l3 = efficientformer_l3_feat(pretrained="checkpoints/efficientformer_l3_300d.pth")
# l7 = efficientformer_l7_feat(pretrained="checkpoints/efficientformer_l7_300d.pth")

h, w = 1080, 1920
def export(torch_model, name: str):

    example_input = torch.rand(1, 3, h, w)
    traced_model = torch.jit.trace(torch_model, example_input)
    out = traced_model(example_input)
    model = ct.convert(
        traced_model,
        convert_to="neuralnetwork",  # mlpackage is very slow dunno why
        inputs=[ct.ImageType(shape=example_input.shape)],
    )
    model.save(name)
    return model

from PIL import Image
model = export(l3, name="l3full_hd.mlmodel")
print("incoming")

arange = 12
total_time = 0
import numpy as np

model.predict({
    'x_1': Image.fromarray((np.random.rand(h, w, 3).astype(np.float32) * 255).astype(np.uint8)),
})

for i in range(arange):
    start = time()
    prediction = model.predict({
        'x_1': Image.fromarray((np.random.rand(h, w, 3).astype(np.float32) * 255).astype(np.uint8)),
    })
    total_time += time() - start
    print("OUT")

print(total_time)
print(total_time/arange)


# f1 full hd => 110ms m1 ipad pro
# f3 224x224 => 6.4ms m1 ipad pro
