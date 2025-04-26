from model import RecyclingResNet as TheModel
from dataset import create_dataloader as the_dataloader


from config import batch_size as the_batch_size
from config import epochs as total_epochs

from config import learning_rate, train_data_path, val_data_path, weight_decay
import torch
from torch import nn, optim

def train_model(model, num_epochs, train_loader, val_loader, loss_fn, optimizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2, factor=0.5, verbose=True)
    
    # make the directory if it doesn't exist
    import os
    if not os.path.exists("_checkpoints"):
        os.makedirs("_checkpoints")
    
    best_val_acc = 0
    
    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        train_loss, train_correct = 0, 0
        
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.float().to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # Metrics Calculation
            train_loss += loss.item()
            preds = (outputs > 0.5).float()
            train_correct += (preds == targets).sum().item()
            
            # Log every 20 batches
            if batch_idx % 20 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
        # Validation Phase
        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.float().to(device)
                outputs = model(data)
                val_loss += loss_fn(outputs, targets).item()
                val_correct += ((outputs > 0.5).float() == targets).sum().item()
        
        # Epoch Statistics
        train_loss /= len(train_loader)
        train_acc = train_correct / len(train_loader.dataset)
        val_loss /= len(val_loader)
        val_acc = val_correct / len(val_loader.dataset)
        
        # Update scheduler
        scheduler.step(val_acc)
        
        # Save Best Model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "_checkpoints/final_weights.pth")
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}\n")

# if __name__ == "__main__":
#     # Initialize components
#     model = TheModel()
#     train_loader = the_dataloader(train_data_path, batch_size=the_batch_size)
#     val_loader = the_dataloader(val_data_path, batch_size=the_batch_size, shuffle=False)
    
#     # Loss and Optimizer with Weight Decay
#     loss_fn = nn.BCELoss()
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
#     # Start training
#     train_model(
#         model=model,
#         num_epochs=total_epochs,
#         train_loader=train_loader,
#         val_loader=val_loader,
#         loss_fn=loss_fn,
#         optimizer=optimizer
#     )
