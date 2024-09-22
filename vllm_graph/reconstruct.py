from torch_mlir import torchscript


def _construct(region):
    for block in region.blocks:
        for operation in block.operations:
            if len(operation.regions) != 0:
                for sub_region in operation.regions:
                    _construct(sub_region)

            else:
                print(operation.name)

def _get_model(module):
    region = module.body.region
    model = _construct(region)
    return model

def reconstruct(model, inputs):

    module = torchscript.compile(model, inputs, output_type=torchscript.OutputType.TORCH)
    new_model = _get_model(module)
    return new_model

