import torch

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    running_loss = 0.0
    n_batches = 0

    for batch in loader:
        images = batch["image"].to(device)
        x_table = batch["x_table"].to(device)
        mask_table = batch["mask_table"].to(device)

        pred = model(images)
        loss = loss_fn(pred, x_table, mask_table)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item())
        n_batches += 1

    return running_loss / max(n_batches, 1)
