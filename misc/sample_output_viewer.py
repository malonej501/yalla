# Visualize a grid of image thumbnails with filenames
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# === Config ===
folder = '../run/sweep_adv_01-06-25'
thumb_size = (100, 100)
cols = 10  # Number of images per row
font_size = thumb_size[0] // 5

# === Load and prepare thumbnails ===
image_files = [f for f in os.listdir(folder) if f.lower().endswith(
    ('jpg', 'jpeg', 'png', 'gif', 'bmp'))]
thumbs = []

# Try to load a default font (fallback included)
try:
    font = ImageFont.truetype("arial.ttf", font_size)
except:
    font = ImageFont.load_default()

for fname in image_files:
    try:
        img_path = os.path.join(folder, fname)
        img = Image.open(img_path).convert('RGB')
        img.thumbnail(thumb_size)

        # Create a new image with space for the filename text
        new_img = Image.new(
            'RGB', (thumb_size[0], thumb_size[1] + 15), 'white')
        new_img.paste(img, (0, 15))

        # Draw filename text
        draw = ImageDraw.Draw(new_img)
        text = fname[:15] + "..." if len(fname) > 18 else fname
        text_width = draw.textlength(text, font=font)
        draw.text(((thumb_size[0] - text_width) / 2, 0),
                  text, font=font, fill='black')

        thumbs.append(new_img)
    except Exception as e:
        print(f"Error processing {fname}: {e}")

# === Assemble into grid ===
rows = (len(thumbs) + cols - 1) // cols
cell_w, cell_h = thumbs[0].size
grid_img = Image.new('RGB', (cols * cell_w, rows * cell_h), 'white')

for i, img in enumerate(thumbs):
    x = (i % cols) * cell_w
    y = (i // cols) * cell_h
    grid_img.paste(img, (x, y))

# === Display result ===
plt.figure(figsize=(12, 12 * (rows / cols)))
plt.imshow(grid_img)
plt.axis('off')
plt.tight_layout()
plt.savefig(f"{folder}/image_grid.pdf", dpi=600)
plt.show()
