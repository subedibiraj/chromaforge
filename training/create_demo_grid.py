import os
import torch
import torchvision
from PIL import Image
import numpy as np
from torchvision import transforms
from models.generator import UNetGenerator
from models.utils import lab_tensor_to_rgb

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model
    model = UNetGenerator().to(device)
    state_dict = torch.load("training/runs/generator_best_clean.pth", map_location=device, weights_only=True)
    if "model_state_dict" in state_dict:
        model.load_state_dict(state_dict["model_state_dict"])
    else:
        model.load_state_dict(state_dict)
    model.eval()

    data_dir = "data/train2017"
    if not os.path.exists(data_dir):
        print("Data dir not found.")
        return

    # Select 4 visually distinct images (hardcode indices for speed or just take first 4)
    image_files = sorted(os.listdir(data_dir))[:4]
    
    from skimage.color import rgb2lab
    
    to_tensor = transforms.ToTensor()
    
    all_images = [] # [gray, colorized, original]

    with torch.no_grad():
        for filename in image_files:
            img_path = os.path.join(data_dir, filename)
            try:
                img = Image.open(img_path).convert("RGB").resize((256, 256), Image.LANCZOS)
            except:
                continue
            img_np = np.array(img, dtype=np.float32) / 255.0
            lab = rgb2lab(img_np).astype(np.float32)
            
            L = torch.from_numpy((lab[:, :, 0:1] / 50.0) - 1.0).permute(2, 0, 1).unsqueeze(0).to(device)
            
            fake_ab = model(L)
            
            fake_rgb = lab_tensor_to_rgb(L[0], fake_ab[0])
            fake_rgb_tensor = torch.from_numpy(fake_rgb).permute(2, 0, 1) / 255.0
            
            gray_rgb_np = np.stack([lab[:, :, 0], lab[:, :, 0], lab[:, :, 0]], axis=-1) / 100.0
            gray_rgb_tensor = torch.from_numpy(gray_rgb_np).permute(2, 0, 1)
            
            real_rgb_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1) / 255.0
            
            all_images.extend([gray_rgb_tensor, fake_rgb_tensor, real_rgb_tensor])
            
    grid = torchvision.utils.make_grid(all_images, nrow=3, padding=4, pad_value=1.0)
    grid_img = transforms.ToPILImage()(grid)
    
    os.makedirs("docs/figures", exist_ok=True)
    grid_img.save("docs/figures/demo_grid.png")
    print("Saved docs/figures/demo_grid.png")

if __name__ == "__main__":
    main()
