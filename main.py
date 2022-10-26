import glob
import logging
import os
from pathlib import Path
import shutil
import sys
import tempfile

import nibabel as nib
import numpy as np
from monai.config import print_config
from monai.data import ArrayDataset, create_test_image_3d, decollate_batch, DataLoader
from monai.handlers import (
    MeanDice,
    MLFlowHandler,
    StatsHandler,
    TensorBoardImageHandler,
    TensorBoardStatsHandler,
)
from monai.losses import DiceLoss
from monai.networks.nets import UNet
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    LoadImage,
    RandSpatialCrop,
    Resize,
    ScaleIntensity,
)
from monai.utils import first

import ignite
import torch

from model import Gibbs_UNet

directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = tempfile.mkdtemp() if directory is None else directory

# create synthesized data
def random_3d_images():
    for i in range(40):
        im, seg = create_test_image_3d(128, 128, 128, num_seg_classes=1)

        n = nib.Nifti1Image(im, np.eye(4))
        nib.save(n, os.path.join(root_dir, 'random', f"im{i}.nii.gz"))

        n = nib.Nifti1Image(seg, np.eye(4))
        nib.save(n, os.path.join(root_dir, 'random', f"seg{i}.nii.gz"))
    
    images = sorted(glob.glob(os.path.join(root_dir, 'random', "im*.nii.gz")))
    segs = sorted(glob.glob(os.path.join(root_dir, 'random', "seg*.nii.gz")))
    return images, segs 

def read_3d_images(images_path, segs_path):
    images = glob.glob(os.path.join(root_dir, images_path, '*'))
    segs = glob.glob(os.path.join(root_dir, segs_path, '*'))
    return images, segs 

def train_val_loop(images, segs):
    print_config()
    print(root_dir)
    # Define transforms for image and segmentation
    imtrans = Compose(
        [
            LoadImage(image_only=True),
            ScaleIntensity(),
            EnsureChannelFirst(),
            RandSpatialCrop((96, 96, 96), random_size=False),
        ]
    )
    segtrans = Compose(
        [
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            RandSpatialCrop((96, 96, 96), random_size=False),
        ]
    )

    # Define nifti dataset, dataloader
    ds = ArrayDataset(images, imtrans, segs, segtrans)
    loader = DataLoader(
        ds, batch_size=10, num_workers=2, pin_memory=torch.cuda.is_available()
    )
    im, seg = first(loader)
    print(im.shape, seg.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Gibbs_UNet(1.0)
    loss = DiceLoss(sigmoid=True)
    lr = 1e-3
    opt = torch.optim.Adam(net.parameters(), lr)

    trainer = ignite.engine.create_supervised_trainer(
        net, opt, loss, device, False
    )
    
    # optional section for checkpoint and tensorboard logging
    # adding checkpoint handler to save models (network
    # params and optimizer stats) during training
    log_dir = os.path.join(root_dir, "logs")
    checkpoint_handler = ignite.handlers.ModelCheckpoint(
        log_dir, "net", n_saved=10, require_empty=False
    )
    trainer.add_event_handler(
        event_name=ignite.engine.Events.EPOCH_COMPLETED,
        handler=checkpoint_handler,
        to_save={"net": net, "opt": opt},
    )

    # StatsHandler prints loss at every iteration
    # user can also customize print functions and can use output_transform to convert
    # engine.state.output if it's not a loss value
    train_stats_handler = StatsHandler(name="trainer", output_transform=lambda x: x)
    train_stats_handler.attach(trainer)

    # TensorBoardStatsHandler plots loss at every iteration
    train_tensorboard_stats_handler = TensorBoardStatsHandler(log_dir=log_dir, output_transform=lambda x: x)
    train_tensorboard_stats_handler.attach(trainer)

    # MLFlowHandler plots loss at every iteration on MLFlow web UI
    mlflow_dir = os.path.join(log_dir, "mlruns")
    train_mlflow_handler = MLFlowHandler(tracking_uri=Path(mlflow_dir).as_uri(), output_transform=lambda x: x)
    train_mlflow_handler.attach(trainer)
    
    # optional section for model validation during training
    validation_every_n_epochs = 1
    # Set parameters for validation
    metric_name = "Mean_Dice"
    # add evaluation metric to the evaluator engine
    val_metrics = {metric_name: MeanDice()}
    post_pred = Compose(
        [Activations(sigmoid=True), AsDiscrete(threshold=0.5)]
    )
    post_label = Compose([AsDiscrete(threshold=0.5)])
    # Ignite evaluator expects batch=(img, seg) and
    # returns output=(y_pred, y) at every iteration,
    # user can add output_transform to return other values
    evaluator = ignite.engine.create_supervised_evaluator(
        net,
        val_metrics,
        device,
        True,
        output_transform=lambda x, y, y_pred: (
            [post_pred(i) for i in decollate_batch(y_pred)],
            [post_label(i) for i in decollate_batch(y)]
        ),
    )

    # create a validation data loader
    val_imtrans = Compose(
        [
            LoadImage(image_only=True),
            ScaleIntensity(),
            EnsureChannelFirst(),
            Resize((96, 96, 96)),
        ]
    )
    val_segtrans = Compose(
        [
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            Resize((96, 96, 96)),
        ]
    )
    val_ds = ArrayDataset(images[-10::], val_imtrans, segs[-10::], val_segtrans)
    val_loader = DataLoader(
        val_ds, batch_size=5, num_workers=8, pin_memory=torch.cuda.is_available()
    )


    @trainer.on(
        ignite.engine.Events.EPOCH_COMPLETED(every=validation_every_n_epochs)
    )
    def run_validation(engine):
        evaluator.run(val_loader)


    # Add stats event handler to print validation stats via evaluator
    val_stats_handler = StatsHandler(
        name="evaluator",
        # no need to print loss value, so disable per iteration output
        output_transform=lambda x: None,
        # fetch global epoch number from trainer
        global_epoch_transform=lambda x: trainer.state.epoch,
    )
    val_stats_handler.attach(evaluator)

    # add handler to record metrics to TensorBoard at every validation epoch
    val_tensorboard_stats_handler = TensorBoardStatsHandler(
        log_dir=log_dir,
        # no need to plot loss value, so disable per iteration output
        output_transform=lambda x: None,
        # fetch global epoch number from trainer
        global_epoch_transform=lambda x: trainer.state.epoch,
    )
    val_tensorboard_stats_handler.attach(evaluator)

    # add handler to record metrics to MLFlow at every validation epoch
    val_mlflow_handler = MLFlowHandler(
        tracking_uri=Path(mlflow_dir).as_uri(),
        # no need to plot loss value, so disable per iteration output
        output_transform=lambda x: None,
        # fetch global epoch number from trainer
        global_epoch_transform=lambda x: trainer.state.epoch,
    )
    val_mlflow_handler.attach(evaluator)

    # add handler to draw the first image and the corresponding
    # label and model output in the last batch
    # here we draw the 3D output as GIF format along Depth
    # axis, at every validation epoch
    val_tensorboard_image_handler = TensorBoardImageHandler(
        log_dir=log_dir,
        batch_transform=lambda batch: (batch[0], batch[1]),
        output_transform=lambda output: output[0],
        global_iter_transform=lambda x: trainer.state.epoch,
    )
    evaluator.add_event_handler(
        event_name=ignite.engine.Events.EPOCH_COMPLETED,
        handler=val_tensorboard_image_handler,
    )
    
    # create a training data loader
    train_ds = ArrayDataset(images, imtrans, segs, segtrans)
    train_loader = DataLoader(
        train_ds,
        batch_size=5,
        shuffle=True,
        num_workers=8,
        pin_memory=torch.cuda.is_available(),
    )
    max_epochs = 5
    state = trainer.run(train_loader, max_epochs)

if __name__ == "__main__":
    # images, segs = read_3d_images('small_BrainTumour_data/imagesTr', 'small_BrainTumour_data/labelsTr')
    if len(sys.argv[1:]) == 2:
      images, segs = read_3d_images(sys.argv[1], sys.argv[2])
    else:
      images, segs = random_3d_images()
    train_val_loop(images, segs)