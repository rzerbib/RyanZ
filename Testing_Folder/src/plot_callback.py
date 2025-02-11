from pytorch_lightning.callbacks import Callback
import matplotlib.pyplot as plt

class PlotLossCallback(Callback):
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.epochs = []

    def on_train_epoch_end(self, trainer, pl_module):
        # Log training loss
        train_loss = trainer.callback_metrics["train_loss_epoch"].cpu().item()
        self.train_losses.append(train_loss)
        self.epochs.append(trainer.current_epoch)

        # If validation loss is available, log it
        if "val_loss" in trainer.callback_metrics:
            val_loss = trainer.callback_metrics["val_loss"].cpu().item()
            self.val_losses.append(val_loss)

        # Plot losses
        plt.figure()
        plt.plot(self.epochs, self.train_losses, label="Training Loss")
        if self.val_losses:
            plt.plot(self.epochs, self.val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"loss_epoch_{trainer.current_epoch}.png")
        plt.close()
