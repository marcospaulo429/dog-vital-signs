import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from pytorch_lightning import LightningModule
from typing import Tuple
from loss import DiscriminatorLoss, GeneratorLoss
from gan import GanomalyModel

class GanomalyMNIST(LightningModule):
    def __init__(
        self,
        input_size: Tuple[int, int] = (32, 32),
        batch_size: int = 32,
        n_features: int = 64,
        latent_vec_size: int = 100,
        extra_layers: int = 0,
        add_final_conv_layer: bool = True,
        wadv: int = 1,
        wcon: int = 50,
        wenc: int = 1,
        lr: float = 0.0002,
        beta1: float = 0.5,
        beta2: float = 0.999,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Model configuration
        self.input_size = input_size
        self.batch_size = batch_size
        self.n_features = n_features
        self.latent_vec_size = latent_vec_size
        self.extra_layers = extra_layers
        self.add_final_conv_layer = add_final_conv_layer
        
        # Labels
        self.register_buffer('real_label', torch.ones(batch_size, dtype=torch.float32))
        self.register_buffer('fake_label', torch.zeros(batch_size, dtype=torch.float32))
        
        # Score tracking
        self.register_buffer('min_scores', torch.tensor(float('inf')))
        self.register_buffer('max_scores', torch.tensor(float('-inf')))
        
        # Initialize model
        self.model = GanomalyModel(
            input_size=input_size,
            num_input_channels=1,  # MNIST has 1 channel
            n_features=n_features,
            latent_vec_size=latent_vec_size,
            extra_layers=extra_layers,
            add_final_conv_layer=add_final_conv_layer
        )
        
        # Loss functions
        self.generator_loss = GeneratorLoss(wadv, wcon, wenc)
        self.discriminator_loss = DiscriminatorLoss()
        
        # Manual optimization required for GAN training
        self.automatic_optimization = False

    def forward(self, x):
        return self.model(x)

    def prepare_data(self):
        MNIST(root='./data', train=True, download=True)
        MNIST(root='./data', train=False, download=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            mnist_full = MNIST(root='./data', train=True, transform=ToTensor())
            normal_idx = mnist_full.targets == 0  # digit 0 as normal class
            self.train_dataset = torch.utils.data.Subset(mnist_full, torch.where(normal_idx)[0])
            self.val_dataset = MNIST(root='./data', train=False, transform=ToTensor())

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def configure_optimizers(self):
        opt_d = optim.Adam(
            self.model.discriminator.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.beta1, self.hparams.beta2)
        )
        opt_g = optim.Adam(
            self.model.generator.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.beta1, self.hparams.beta2)
        )
        return [opt_d, opt_g]

    def training_step(self, batch, batch_idx):
        # Get both optimizers
        opt_d, opt_g = self.optimizers()
        
        images, _ = batch
        
        # Forward pass
        padded, fake, latent_i, latent_o = self.model(images)
        pred_real, _ = self.model.discriminator(padded)
        
        ##########################
        # Train Discriminator
        ##########################
        pred_fake, _ = self.model.discriminator(fake.detach())
        d_loss = self.discriminator_loss(pred_real, pred_fake)
        
        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()
        
        ##########################
        # Train Generator
        ##########################
        pred_fake, _ = self.model.discriminator(fake)
        g_loss = self.generator_loss(latent_i, latent_o, padded, fake, pred_real, pred_fake)
        
        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()
        
        # Logging
        self.log_dict({
            'd_loss': d_loss,
            'g_loss': g_loss,
            'g_adv_loss': g_loss.adv_loss,
            'g_con_loss': g_loss.con_loss,
            'g_enc_loss': g_loss.enc_loss,
        }, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        _, _, latent_i, latent_o = self.model(images)
        scores = torch.mean(torch.pow(latent_i - latent_o, 2), dim=1)
        
        # Update min/max scores
        self.min_scores = torch.minimum(self.min_scores, scores.min())
        self.max_scores = torch.maximum(self.max_scores, scores.max())
        
        return {'scores': scores, 'labels': labels}

    def on_validation_epoch_end(self):
        # You could add anomaly score thresholding here
        pass


from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

model = GanomalyMNIST(
    input_size=(32, 32),
    batch_size=128,
    n_features=64,
    latent_vec_size=100
)

logger = TensorBoardLogger("tb_logs", name="ganomaly_mnist_v2")
trainer = Trainer(
    max_epochs=50,
    accelerator='auto',
    logger=logger,
    log_every_n_steps=10
)

trainer.fit(model)