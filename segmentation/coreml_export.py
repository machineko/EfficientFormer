from time import time

import coremltools as ct
from backbone import efficientformer_l1_feat, efficientformer_l3_feat, efficientformer_l7_feat
import torch
from coreml_torch_utils import InputEnumeratedShapeImage, RenameOutput, OutputStaticImage, CoreExporter

l1 = efficientformer_l1_feat(pretrained="checkpoints/efficientformer_l1_1000d.pth")
# l3 = efficientformer_l3_feat(pretrained="checkpoints/efficientformer_l3_300d.pth")
# l7 = efficientformer_l7_feat(pretrained="checkpoints/efficientformer_l7_300d.pth")

h, w = 1024, 2048
def export(torch_model, name: str):

    example_input = torch.rand(1, 3, h, w)
    traced_model = torch.jit.trace(torch_model, example_input)
    out = traced_model(example_input)
    model = ct.convert(
        traced_model,
        convert_to="neuralnetwork",  # mlpackage is very slow dunno why
        inputs=[ct.ImageType(shape=example_input.shape)],
    )
    exporter = CoreExporter(
        utils_methods=[
            RenameOutput(f"stage_{i}_out", output_idx=i-1) for i in range(1, 5)
        ]
    )
    model = exporter(model)
    model.save(name)
    return model

from PIL import Image
model = export(l1, name="l1_1024x2048_nearest.mlmodel")
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


# f1 full hd => 110ms m1 ipad pro bilinear interpolate
# f3 224x224 => 6.4ms m1 ipad pro
# f3 full hd => 339ms m1 ultra 48core gpu // performance report not working yet on macos13 using python so for sure a bit faster in pure swift
# f3 full hd => 320ms m1 ipad pro 8gb from performance report [8 gpu ops all upsample ops are on gpu]
# f3 512x512 => 20ms m1 ultra on nearest upscale
# f3 512x512 => 20ms m1 ipad pro on nearest upscale all ops on ANE :)

# f1 full hd (1024x2048) => 150ms m1 ultra nearest
# f1 full hd (1024x2048) => 110ms ipad pro bilinear interpolate
