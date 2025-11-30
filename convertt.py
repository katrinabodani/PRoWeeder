import torch
from safetensors.torch import load_file
import os

print("="*60)
print("CONVERTING FOLD 5 CHECKPOINT")
print("="*60)

# Path to Fold 5 safetensors
safetensors_path = r"wandb\offline-run-20251129_234353-ft8n1nvr\files\best\model.safetensors"

# Check file exists
if not os.path.exists(safetensors_path):
    print(f"❌ ERROR: File not found at {safetensors_path}")
    exit()

print(f"\n📂 Loading: {safetensors_path}")

# Load safetensors
state_dict = load_file(safetensors_path)
print(f"✅ Loaded {len(state_dict)} parameters")

# Create checkpoint
checkpoint = {
    'model_state_dict': state_dict,
    'epoch': 27,
    'best_f1': 0.8153,  # Validation F1
    'test_f1': 0.7253,  # Test F1
    'fold': 5
}

# Create output directory
os.makedirs(r"weedmap-inference\models\reproduced_roweeder", exist_ok=True)

# Save as .pth
output_path = r"weedmap-inference\models\reproduced_roweeder\best_checkpoint.pth"
torch.save(checkpoint, output_path)

# Verify
size_mb = os.path.getsize(output_path) / 1024 / 1024

print(f"\n✅ CONVERSION COMPLETE!")
print(f"📁 Saved to: {output_path}")
print(f"📦 Size: {size_mb:.1f} MB")
print(f"📊 Validation F1: 81.53%")
print(f"📊 Test F1: 72.53%")
print(f"🏆 Fold: 5/5")
print("="*60)

exit()