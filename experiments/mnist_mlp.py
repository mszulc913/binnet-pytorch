from typing import Sequence, Tuple, List

import torch.nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST  # type: ignore
from torchvision.transforms import transforms  # type: ignore

from binnet.layers import BinaryLinear


class BinMLPClassifier(pl.LightningModule):
    def __init__(
        self,
        in_features: int,
        hidden_sizes: Sequence[int],
        n_classes: int,
        learning_rate: float,
        use_xnor: bool,
    ):
        """Multi Layer Perceptron Classifier with binarized hidden layers.

        First and last layers are not binarized because it may greatly
        lower the performance.

        :param in_features: Number of input features.
        :param hidden_sizes: List of sizes of hidden layers.
        :param n_classes: Number of classes to predict.
        :param learning_rate: Learning rate of ADAM optimizer.
        :param use_xnor: True to use custom XNOR CUDA kernel.
        """
        if len(hidden_sizes) == 0:
            raise ValueError(
                "You need to specify at least one hidden layer in `hidden_sizes` list."
            )

        super().__init__()

        hidden_and_activations: List[torch.nn.Module] = []
        for prev_size, size in zip(hidden_sizes, hidden_sizes[1:]):
            hidden_and_activations.append(
                BinaryLinear(prev_size, size, use_xnor_kernel=use_xnor)
            )
            hidden_and_activations.append(torch.nn.Hardtanh())

        self.input_proj = torch.nn.Linear(in_features, hidden_sizes[0])
        self.hidden = torch.nn.ModuleList(hidden_and_activations)
        self.output_proj = torch.nn.Linear(hidden_sizes[-1], n_classes)

        self.loss = torch.nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

        self.save_hyperparameters()

        self.accuracy = torchmetrics.Accuracy()

    def forward(  # type: ignore
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.input_proj(x)
        x = F.hardtanh(x)
        for layer in self.hidden:
            x = layer.forward(x)
        x = self.output_proj(x)
        x = x.squeeze(-2)
        return x, torch.argmax(x, dim=-1)

    def training_step(  # type: ignore
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        probs, preds = self.forward(x)
        self.accuracy(preds, y)
        self.log("acc", self.accuracy, prog_bar=True)
        return self.loss(probs, y)

    def test_step(  # type: ignore
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ):
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
    batch_size = 100
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            torch.nn.Flatten(0, -1),
        ]
    )
    mnist_train = MNIST("", train=True, download=True, transform=transform)
    mnist_test = MNIST("", train=False, download=True, transform=transform)

    train_loader = DataLoader(mnist_train, batch_size=batch_size, num_workers=4)
    test_loader = DataLoader(mnist_test, batch_size=batch_size, num_workers=4)

    model = BinMLPClassifier(28 * 28, (4096, 4096, 4096), 10, 1e-4, True)

    trainer = pl.Trainer(max_epochs=1, gpus=1)
    trainer.fit(model, train_loader)

    trainer.test(dataloaders=test_loader)


if __name__ == "__main__":
    cli_main()
