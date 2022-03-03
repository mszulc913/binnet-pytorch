from typing import Sequence, Tuple

import torch.nn
import pytorch_lightning as pl
import torchmetrics
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from layers import BinaryLinear


class BinMLPClassifier(pl.LightningModule):
    def __init__(
        self,
        in_features: int,
        hidden_sizes: Sequence[int],
        n_classes: int,
        learning_rate: float,
    ):
        """Multi Layer Perceptron Classifier with binarized hidden layers.

        First and last layers are not binarized.

        :param in_features: Number of input features.
        :param hidden_sizes: List of sizes of hidden layers.
        :param n_classes: Number of classes to predict.
        :param learning_rate: Learning rate of ADAM optimizer.
        """
        if len(hidden_sizes) == 0:
            raise ValueError(
                "You need to specify at least one hidden layer in `hidden_sizes` list."
            )

        super().__init__()

        hidden_and_activations = []
        for prev_size, size in zip(hidden_sizes, hidden_sizes[1:]):
            hidden_and_activations.append(BinaryLinear(prev_size, size))
            hidden_and_activations.append(torch.nn.Hardtanh())

        self.hidden = torch.nn.ModuleList(
            (
                torch.nn.Linear(in_features, hidden_sizes[0]),
                torch.nn.Hardtanh(),
                *hidden_and_activations,
                torch.nn.Linear(hidden_sizes[-1], n_classes),
            )
        )
        self.loss = torch.nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

        self.save_hyperparameters()

        self.accuracy = torchmetrics.Accuracy()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for layer in self.hidden:
            x = layer.forward(x)
        x = x.squeeze(-2)
        return x, torch.argmax(x, dim=-1)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        probs, preds = self.forward(x)
        self.accuracy(preds, y)
        self.log("acc", self.accuracy, prog_bar=True)
        return self.loss(probs, y)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        probs, preds = self.forward(x)
        loss = self.loss(probs, y)
        self.accuracy(preds, y)
        self.log("test_loss", loss)

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.log("test_acc", self.accuracy)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


def cli_main():
    pl.seed_everything(1234)
    batch_size = 256

    mnist_train = MNIST(
        "",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), torch.nn.Flatten()]),
    )
    mnist_test = MNIST(
        "",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), torch.nn.Flatten()]),
    )

    train_loader = DataLoader(mnist_train, batch_size=batch_size)
    test_loader = DataLoader(mnist_test, batch_size=batch_size)

    model = BinMLPClassifier(28 * 28, (1024, 1024, 512), 10, 1e-4)

    trainer = pl.Trainer(max_epochs=5, gpus=1)
    trainer.fit(model, train_loader)

    trainer.test(test_dataloaders=test_loader)


if __name__ == "__main__":
    cli_main()
