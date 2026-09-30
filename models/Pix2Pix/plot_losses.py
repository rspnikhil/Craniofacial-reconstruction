import json
import matplotlib.pyplot as plt
import os

def plot_loss_curves(json_path="loss_history.json", save_path="loss_curves.png"):
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found. Please train the model first to generate loss history.")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)
        
    loss_G = data.get("loss_G", [])
    loss_D = data.get("loss_D", [])
    
    epochs = range(1, len(loss_G) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss_G, label='Generator Loss (Total)', color='blue', linewidth=2)
    plt.plot(epochs, loss_D, label='Discriminator Loss', color='orange', linewidth=2)
    
    plt.title('Pix2Pix Training Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Loss curves successfully saved to {save_path}")
    
if __name__ == "__main__":
    plot_loss_curves()
