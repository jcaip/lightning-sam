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

        # from torch.ao.pruning import WeightNormSparsifier
        if self.cfg.model.sparse:
            print("Enabling sparse aware training")
            if self.cfg.model.sparse.fast_sparse_training:
            from torchao.sparsity.prototype.fast_sparse_training import swap_linear_with_semi_sparse_linear_
            sparse_config = []
            for name, mod in self.model.named_modules():
                if isinstance(mod, torch.nn.Linear) and 'image_encoder' in name and 'mlp' in name:
                    # sparse_config.append({"tensor_fqn": f"{name}.weight"})
                    # mod.weight.data.to(torch.bfloat16, inplace=True) = mod.weight.data.to(torch.bfloat16)
                    sparse_config.append(name)

            # pprint(sparse_config)
            swap_linear_with_semi_sparse_linear_(self.model, sparse_config)
            # sparsifier = WeightNormSparsifier(
            #     sparsity_level=2.0, sparse_block_shape=(1, 4), zeros_per_block=2
            # )
            # sparsifier.prepare(model, sparse_config)
            # sparsifier.step()
            self.model.image_encoder = torch.compile(self.model.image_encoder, mode='max-autotune')

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
