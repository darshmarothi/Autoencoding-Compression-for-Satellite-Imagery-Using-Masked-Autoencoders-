import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as T
from einops import rearrange
import base64
import io

# ============================================================
# SAME MODEL ARCHITECTURE YOU TRAINED IN YOUR NOTEBOOK
# ============================================================

class PatchEmbed(nn.Module):
    def _init_(self, img_size=128, patch_size=16, in_ch=3, embed_dim=256):
        super()._init_()
        self.patch_size = patch_size
        self.img_size = img_size
        # we keep num_patches here for clarity but PatchEmbed doesn't need to enforce it
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B,3,H,W) -> (B,embed,h',w') -> (B, N, embed)
        x = self.proj(x)
        return rearrange(x, "b c h w -> b (h w) c")


class MAELightCorrected(nn.Module):
    def _init_(self, img_size=128, patch_size=16, embed_dim=256,
                 depth=6, decoder_depth=4, mask_ratio=0.75):
        super()._init_()
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio

        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # Use batch_first=True for clearer shapes and to avoid warnings
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8,
                                                   dim_feedforward=embed_dim * 2,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        decoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8,
                                                   dim_feedforward=embed_dim * 2,
                                                   batch_first=True)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=decoder_depth)

        self.decoder_pred = nn.Linear(embed_dim, patch_size * patch_size * 3)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, imgs):
        """
        imgs: (B,3,H,W)
        returns:
          pred: (B, N, patch_dim)  -> decoded predictions for all patches
          latent: (B, num_keep, D)
        """
        B = imgs.shape[0]
        x = self.patch_embed(imgs) + self.pos_embed  # (B,N,D)

        N, D = x.shape[1], x.shape[2]
        num_keep = int(N * (1 - self.mask_ratio))

        # random mask
        noise = torch.rand(B, N, device=imgs.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :num_keep]

        # gather visible tokens
        x_keep = torch.gather(x, 1, ids_keep.unsqueeze(-1).repeat(1, 1, D))  # (B, num_keep, D)
        latent = self.encoder(self.norm(x_keep))  # (B, num_keep, D)

        # prepare decoder input (encoded visible + mask tokens)
        mask_tokens = self.mask_token.repeat(B, N - num_keep, 1)
        x_full = torch.cat([latent, mask_tokens], dim=1)  # (B, N, D) in shuffled order
        x_full = torch.gather(x_full, 1, ids_restore.unsqueeze(-1).repeat(1, 1, D))  # restore order
        x_full = x_full + self.pos_embed

        dec = self.decoder(self.norm(x_full))  # (B, N, D)
        pred = self.decoder_pred(dec)  # (B, N, patch_dim)

        return pred, latent


def patches_to_img(patches, patch_size, nside):
    """
    patches: (B, N, c*p*p)
    returns: (B, c, H, W)
    """
    return rearrange(
        patches,
        "b (h w) (c p1 p2) -> b c (h p1) (w p2)",
        h=nside, w=nside, c=3, p1=patch_size, p2=patch_size
    )


# ============================================================
#             LOAD TRAINED MODEL
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

model = MAELightCorrected()
# try loading model checkpoint
try:
    raw = torch.load("mae_light_corrected_final.pth", map_location=device)
except FileNotFoundError:
    st.error("Model file 'mae_light_corrected_final.pth' not found in the workspace. Put the .pth file next to app.py")
    st.stop()

# Support different checkpoint formats
if isinstance(raw, dict) and "state_dict" in raw:
    state_dict = raw["state_dict"]
else:
    state_dict = raw

# Clean keys if saved from DataParallel
clean_state = {}
for k, v in state_dict.items():
    new_k = k.replace("module.", "") if k.startswith("module.") else k
    clean_state[new_k] = v

# Load weights (try strict=True then fallback)
try:
    model.load_state_dict(clean_state, strict=True)
except RuntimeError as e:
    # fallback (this will still work in many common cases)
    model.load_state_dict(clean_state, strict=False)
    st.warning(f"Model loaded with strict=False due to mismatch: {e}")

model.to(device)
model.eval()

# preprocessing transform (must match training)
transform = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor()
])

# ============================================================
#                   STREAMLIT UI
# ============================================================
st.set_page_config(layout="centered", page_title="MAE Satellite Image Compression")
st.title("📦 MAE Satellite Image Compression App")
st.write("Upload an image → Compress → Reconstruct using your trained MAE")

uploaded_file = st.file_uploader("Upload Satellite Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load image and resize (so model always receives 128x128)
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize((128, 128), Image.BILINEAR)
    st.image(img_resized, caption="Original (resized to 128×128)", width=350)

    # prepare tensor
    img_tensor = transform(img_resized).unsqueeze(0).to(device)

    # forward pass (inference)
    with torch.no_grad():
        pred, latent = model(img_tensor)

    # reconstruct image from pred
    N = pred.shape[1]
    nside = int(np.sqrt(N))
    recon = patches_to_img(pred, patch_size=model.patch_size, nside=nside)  # (B,3,H,W)
    recon_np = recon[0].permute(1, 2, 0).cpu().numpy()
    recon_np = np.clip(recon_np, 0.0, 1.0)

    st.image(recon_np, caption="Reconstructed Image", width=350)

    # ---------------------------
    # Compression size info
    # ---------------------------
    latent_np = latent.detach().cpu().numpy()
    latent_bytes = latent_np.nbytes
    original_size_bytes = (128 * 128 * 3)  # raw RGB bytes (uint8)
    st.subheader("📊 Compression Info")
    st.write(f"Original (raw RGB): *{original_size_bytes/1024:.2f} KB*")
    st.write(f"Compressed latent (float32): *{latent_bytes/1024:.2f} KB*")

    # ---------------------------
    # Download latent as .bin
    # ---------------------------
    latent_bytes_data = latent_np.tobytes()
    b64 = base64.b64encode(latent_bytes_data).decode("utf-8")
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="compressed_latent.bin">⬇ Download Compressed Latent</a>'
    st.markdown(href, unsafe_allow_html=True)

st.write("---")
st.header("🔄 Load Compressed Latent → Decompress")

latent_file = st.file_uploader("Upload latent (.bin) to decode", type=["bin"])

if latent_file:
    bin_data = latent_file.read()
    # We assume float32 storage (as saved from numpy.tobytes())
    latent_arr = np.frombuffer(bin_data, dtype=np.float32)

    # reshape to (1, num_keep, embed_dim)
    N = model.num_patches
    num_keep = int(N * (1 - model.mask_ratio))
    D = model.embed_dim

    try:
        latent_arr = latent_arr.reshape(1, num_keep, D)
    except Exception as e:
        st.error(f"Uploaded latent has unexpected size/shape. Expected shape -> (1, {num_keep}, {D}). Error: {e}")
        st.stop()

    latent_tensor = torch.tensor(latent_arr, dtype=torch.float32).to(device)

    # Build decoder-only input: latent + mask tokens, then restore order
    with torch.no_grad():
        B = 1
        # mask tokens
        mask_tokens = model.mask_token.repeat(B, N - num_keep, 1).to(device)
        # concat latent + mask (this order is the shuffled order in our forward)
        x_full_shuffled = torch.cat([latent_tensor, mask_tokens], dim=1)  # (1, N, D)

        # we need ids_restore to place them back to original order - b_