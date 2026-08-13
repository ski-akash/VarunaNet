# models/

Model architectures, loss functions, and export logic — the "what predicts water" layer.

This folder holds:
- Segmentation architectures compared in the benchmark: U-Net (ResNet-34 encoder), U-Net++/Attention U-Net, DeepLabV3+, and SegFormer.
- Loss functions used for training (Dice + BCE combo, with Focal loss as an ablation), including handling of the "ignore" label for no-data pixels.
- Export code to convert a trained PyTorch model to ONNX, so it can be served without needing PyTorch installed.

This folder defines the models but doesn't run the training loop itself — that lives in `training/`.
