import torch.nn as nn
import torch.nn.functional as F
from segment_anything_fast import sam_model_registry
from segment_anything_fast import SamPredictor
import torch

class Model(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self):
        self.model = sam_model_registry[self.cfg.model.type](checkpoint=self.cfg.model.checkpoint)
        self.model.train()
        if self.cfg.model.freeze.image_encoder:
            for param in self.model.image_encoder.parameters():
                param.requires_grad = False
        if self.cfg.model.freeze.prompt_encoder:
            for param in self.model.prompt_encoder.parameters():
                param.requires_grad = False
        if self.cfg.model.freeze.mask_decoder:
            for param in self.model.mask_decoder.parameters():
                param.requires_grad = False

        if self.cfg.model.sparse.enable:
            print("Enabling sparse aware training")

            if self.cfg.model.sparse.type == 'simulated':
                print(self.cfg.model.sparse.type)
                sparse_config = []
                from torch.ao.pruning import WeightNormSparsifier
                for name, mod in self.model.named_modules():
                    if isinstance(mod, torch.nn.Linear) and 'image_encoder' in name: 
                        sparse_config.append({"tensor_fqn": f"{name}.weight"})

                sparsifier = WeightNormSparsifier(
                    sparsity_level=1.0, sparse_block_shape=(1, 4), zeros_per_block=2
                )
                from pprint import pprint
                pprint(sparse_config)
                sparsifier.prepare(self.model, sparse_config)
                sparsifier.step()

            elif self.cfg.model.sparse.type == 'accelerated':
                print(self.cfg.model.sparse.type)
                sparse_config = {}
                from torchao.sparsity.training import swap_linear_with_semi_sparse_linear, SemiSparseLinear
                for name, mod in self.model.named_modules():
                    if isinstance(mod, torch.nn.Linear) and 'image_encoder' in name and 'mlp' in name:
                        mod.weight = nn.Parameter(mod.weight)
                        sparse_config[name] = SemiSparseLinear

                swap_linear_with_semi_sparse_linear(self.model, sparse_config)
                self.model.image_encoder = torch.compile(self.model.image_encoder, mode="max-autotune")
            else:
                raise ValueError(f"Unknown sparse type {self.cfg.model.sparse.type}")
        
        print(self.model)

    def forward(self, images, bboxes):
        _, _, H, W = images.shape
        image_embeddings = self.model.image_encoder(images)
        pred_masks = []
        ious = []
        for embedding, bbox in zip(image_embeddings, bboxes):
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=None,
                boxes=bbox,
                masks=None,
            )

            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=embedding.unsqueeze(0),
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            masks = F.interpolate(
                low_res_masks,
                (H, W),
                mode="bilinear",
                align_corners=False,
            )
            pred_masks.append(masks.squeeze(1))
            ious.append(iou_predictions)

        return pred_masks, ious

    def get_predictor(self):
        return SamPredictor(self.model)
