import torch
from dataset import HumanCatEyesDataset
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator


def train_fn(
    disc_H, gen_C, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler
):
    H_reals = 0
    H_fakes = 0
    loop = tqdm(loader, leave=True)

    for idx, (cat, human) in enumerate(loop):
        cat = cat.to(config.DEVICE)
        human = human.to(config.DEVICE)

        # Train Discriminator H
        with torch.amp.autocast('cuda'):
            fake_cat = gen_C(human)
            D_H_real = disc_H(cat)
            D_H_fake = disc_H(fake_cat.detach())
            H_reals += D_H_real.mean().item()
            H_fakes += D_H_fake.mean().item()
            D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real))
            D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake))
            D_H_loss = D_H_real_loss + D_H_fake_loss


            # Total discriminator loss
            D_loss = D_H_loss

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generator C
        with torch.amp.autocast('cuda'):
            D_H_fake = disc_H(fake_cat)
            loss_G_C = mse(D_H_fake, torch.ones_like(D_H_fake))

            # cycle loss
            cycle_human = gen_C(fake_cat)
            cycle_human_loss = l1(human, cycle_human)

            # Identity loss
            identity_human = gen_C(human)
            identity_human_loss = l1(human, identity_human)

            # Total generator loss
            G_loss = (
                loss_G_C
                + cycle_human_loss * config.LAMBDA_CYCLE
                + identity_human_loss * config.LAMBDA_IDENTITY
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        # Display generated image at the end of each epoch
        if idx % 200 == 0:
            save_image(fake_cat * 0.5 + 0.5, f"C:/Idris/Ecole/2A/CycleGAN/saved_images/cat_{idx}.png")

        loop.set_postfix(H_real=H_reals / (idx + 1), H_fake=H_fakes / (idx + 1))


def main():
    disc_H = Discriminator(in_channels=3).to(config.DEVICE)
    gen_C = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_H.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_C.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_C,
            gen_C,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_H,
            disc_H,
            opt_disc,
            config.LEARNING_RATE,
        )

    dataset = HumanCatEyesDataset(
    root_cat=config.TRAIN_DIR + "/cats", 
    root_human=config.TRAIN_DIR + "/humans",  
    transform=config.transforms,
)

    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    g_scaler = torch.amp.GradScaler(device='cuda')
    d_scaler = torch.amp.GradScaler(device='cuda')

    for epoch in range(config.NUM_EPOCHS):
        train_fn(
            disc_H,
            gen_C,
            loader,
            opt_disc,
            opt_gen,
            L1,
            mse,
            d_scaler,
            g_scaler,
        )

        if config.SAVE_MODEL:
            save_checkpoint(gen_C, opt_gen, filename=config.CHECKPOINT_GEN_C)

if __name__ == "__main__":
    main()
