"""
RoadSense — Deep Cross-Attention Multi-Modal Fusion Network (v2 - Calibrated)
=============================================================================
PyTorch implementation of the Wavelet-Transformer Cross-Attention architecture
fusing multi-axis IMU vibration telemetry with camera-derived defect geometry.

Architecture Upgrades:
1. Dual Wavelet Filterbank (Morlet + Ricker):
   - Ricker (Mexican-Hat) kernels isolate sharp impact impulse shocks (12-25 Hz).
   - Morlet damped sinusoidal kernels capture suspension ringing and rebound decay (6-14 Hz).
2. Physics-Informed Quarter-Car Ground Profile Embedding:
   - 6th input channel: True ground displacement z_road(t) in mm (deconvolved from chassis acceleration).
3. Binary Focal Loss with Logits (gamma=2.0, alpha=0.65):
   - Prioritizes hard ambiguous road defects over thousands of easy smooth-road negatives.
4. Explainable Multi-Modal AI (XAI):
   - Returns cross-attention alignment weights alpha(t).
5. Multi-Task Heads:
   - Binary Hazard Classification, 3-class Severity Grading, and Metric Cavity Depth Regression.
"""

import os
import sys
import math
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score
)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


# ====================================================================
# 1. DUAL WAVELET FILTERBANK (RICKER + MORLET) LAYER
# ====================================================================

class DualWaveletFilterbank(nn.Module):
    """
    Differentiable 1D Dual-Wavelet Filterbank combining:
    - Multi-scale Ricker kernels for impulse impact transients
    - Multi-scale Morlet kernels for suspension ringing & resonance
    """
    def __init__(
        self,
        in_channels=6,
        ricker_scales=[2.0, 4.0, 8.0],
        morlet_scales=[3.0, 6.0, 12.0],
        kernel_size=33
    ):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size

        t = torch.linspace(-4.0, 4.0, kernel_size)
        filters = []

        # Ricker kernels (impulse detection)
        for s in ricker_scales:
            ts = t / (s / 4.0)
            ricker = (1.0 - ts ** 2) * torch.exp(-0.5 * (ts ** 2))
            ricker = ricker / (torch.norm(ricker) + 1e-6)
            filters.append(ricker)

        # Morlet kernels (resonance / oscillation detection)
        for s in morlet_scales:
            ts = t / (s / 4.0)
            morlet = torch.cos(5.0 * ts) * torch.exp(-0.5 * (ts ** 2))
            morlet = morlet / (torch.norm(morlet) + 1e-6)
            filters.append(morlet)

        self.num_scales = len(filters)  # 6 scales total
        weight_tensor = torch.stack(filters).unsqueeze(1)  # [num_scales, 1, kernel_size]
        weight_tensor = weight_tensor.repeat(in_channels, 1, 1)

        self.wavelet_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels * self.num_scales,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=in_channels,
            bias=False
        )
        self.wavelet_conv.weight = nn.Parameter(weight_tensor, requires_grad=True)

    def forward(self, x):
        # x shape: [B, in_channels, T]
        scalogram = self.wavelet_conv(x)
        return F.gelu(scalogram)


# ====================================================================
# 2. SINUSOIDAL POSITIONAL ENCODING
# ====================================================================

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ====================================================================
# 3. BINARY FOCAL LOSS WITH LOGITS
# ====================================================================

class BinaryFocalLossWithLogits(nn.Module):
    """
    Focal Loss for dense class imbalance:
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    Down-weights easy well-classified road examples and focuses on hard bumps.
    """
    def __init__(self, alpha=0.65, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = alpha_t * ((1.0 - p_t) ** self.gamma) * ce_loss
        return loss.mean() if self.reduction == 'mean' else loss.sum()


# ====================================================================
# 4. DEEP CROSS-ATTENTION FUSION NETWORK
# ====================================================================

class DeepCrossAttentionFusionNet(nn.Module):
    """
    Multi-Modal Cross-Attention Transformer fusing physics-informed IMU Wavelet
    scalograms with photogrammetric visual tokens.
    """
    def __init__(
        self,
        imu_channels=6,
        vis_features=6,
        d_model=64,
        nhead=4,
        dim_feedforward=128,
        seq_len=32,
        dropout=0.15
    ):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.imu_channels = imu_channels
        self.vis_features = vis_features

        # 1. Dual Wavelet Filterbank
        self.wavelet_bank = DualWaveletFilterbank(
            in_channels=imu_channels,
            ricker_scales=[2.0, 4.0, 8.0],
            morlet_scales=[3.0, 6.0, 12.0]
        )
        total_imu_feats = imu_channels * self.wavelet_bank.num_scales

        self.imu_projection = nn.Sequential(
            nn.Conv1d(total_imu_feats, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.pos_encoder = SinusoidalPositionalEncoding(d_model, max_len=seq_len)

        # IMU Self-Attention (captures suspension rebound oscillations)
        self.imu_self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.imu_norm1 = nn.LayerNorm(d_model)
        self.imu_ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.imu_norm2 = nn.LayerNorm(d_model)

        # 2. Visual Token Branch
        self.vis_encoder = nn.Sequential(
            nn.Linear(vis_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )

        # 3. Multi-Head Cross-Attention Layer
        # Queries = Visual Candidate Token [B, 1, D]
        # Keys, Values = IMU Wavelet Tokens [B, T, D]
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.cross_norm1 = nn.LayerNorm(d_model)
        self.cross_ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.cross_norm2 = nn.LayerNorm(d_model)

        # 4. Multi-Task Output Heads
        # Head 1: Binary Hazard Classification (Pothole vs Normal)
        self.hazard_classifier = nn.Sequential(
            nn.Linear(d_model * 2, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        # Head 2: Defect Severity Categorization (0: Low, 1: Moderate, 2: Severe)
        self.severity_classifier = nn.Sequential(
            nn.Linear(d_model * 2, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3)
        )

        # Head 3: Metric Cavity Depth (cm) Regression
        self.depth_regressor = nn.Sequential(
            nn.Linear(d_model * 2, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.ReLU()  # Depth is non-negative
        )

    def forward(self, imu_seq, vis_feat, return_attention=False):
        # imu_seq: [B, C=6, T=32]
        # vis_feat: [B, F=6]

        # --- 1. IMU Wavelet Processing ---
        wavelet_feats = self.wavelet_bank(imu_seq)  # [B, C*scales, T]
        imu_tokens = self.imu_projection(wavelet_feats).transpose(1, 2)  # [B, T, D]
        imu_tokens = self.pos_encoder(imu_tokens)

        # Self-attention over IMU time-steps
        imu_sa, _ = self.imu_self_attn(imu_tokens, imu_tokens, imu_tokens)
        imu_tokens = self.imu_norm1(imu_tokens + imu_sa)
        imu_tokens = self.imu_norm2(imu_tokens + self.imu_ffn(imu_tokens))  # [B, T, D]

        # Global IMU pool (mean over time)
        imu_pooled = torch.mean(imu_tokens, dim=1)  # [B, D]

        # --- 2. Visual Token Encoding ---
        vis_token = self.vis_encoder(vis_feat).unsqueeze(1)  # [B, 1, D]

        # --- 3. Cross-Attention: Visual attends to IMU Vibration ---
        cross_out, attn_weights = self.cross_attn(
            query=vis_token,
            key=imu_tokens,
            value=imu_tokens,
            need_weights=True
        )
        cross_token = self.cross_norm1(vis_token + cross_out)
        cross_token = self.cross_norm2(cross_token + self.cross_ffn(cross_token)).squeeze(1)  # [B, D]

        # --- 4. Fusion Representation ---
        fused = torch.cat([cross_token, imu_pooled], dim=-1)  # [B, 2*D]

        # Multi-task outputs
        hazard_logit = self.hazard_classifier(fused).squeeze(-1)  # [B]
        severity_logits = self.severity_classifier(fused)         # [B, 3]
        depth_pred = self.depth_regressor(fused).squeeze(-1)      # [B]

        if return_attention:
            attn_squeezed = attn_weights.squeeze(1)  # [B, T]
            return hazard_logit, severity_logits, depth_pred, attn_squeezed

        return hazard_logit, severity_logits, depth_pred


# ====================================================================
# 5. DATASET & MULTI-MODAL WINDOW BUILDER
# ====================================================================

class RoadSenseMultiModalDataset(Dataset):
    def __init__(self, imu_windows, vis_features, labels, severities, depths):
        self.imu = torch.tensor(imu_windows, dtype=torch.float32)
        self.vis = torch.tensor(vis_features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.severities = torch.tensor(severities, dtype=torch.long)
        self.depths = torch.tensor(depths, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.imu[idx],
            self.vis[idx],
            self.labels[idx],
            self.severities[idx],
            self.depths[idx]
        )


def build_multimodal_windows(df, window_size=32):
    """
    Extracts sliding temporal multi-axis IMU windows and corresponding visual tokens.
    Embeds the 6th physical channel: Deconvolved Ground Elevation Displacement z_road(t) in mm.
    """
    print(f"\n[DATA] Building physics-calibrated multi-modal windows (Window={window_size} frames)...")

    # Import suspension calibrator
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ml_pipeline.suspension_calibrator import SuspensionCalibrator

    df = df.copy()
    accel_z = pd.to_numeric(df['accel_z'], errors='coerce').fillna(9.81).values
    accel_x = pd.to_numeric(df['accel_x'], errors='coerce').fillna(0.0).values if 'accel_x' in df.columns else np.zeros_like(accel_z)
    accel_y = pd.to_numeric(df['accel_y'], errors='coerce').fillna(0.0).values if 'accel_y' in df.columns else np.zeros_like(accel_z)

    # Dynamic acceleration magnitude: |sqrt(ax^2+ay^2+az^2) - 9.81|
    dyn_accel = np.abs(np.sqrt(accel_x**2 + accel_y**2 + accel_z**2) - 9.81)

    # Bandpass & Jerk
    bandpass = pd.to_numeric(df['accel_bandpass'], errors='coerce').fillna(0.0).values if 'accel_bandpass' in df.columns else (accel_z - 9.81)
    jerk = pd.to_numeric(df['jerk'], errors='coerce').fillna(0.0).values if 'jerk' in df.columns else np.gradient(accel_z)
    gyro_y = pd.to_numeric(df['gyro_y'], errors='coerce').fillna(0.0).values if 'gyro_y' in df.columns else np.zeros_like(accel_z)

    # 6th Channel: Physics-informed State-Space Deconvolved Ground Profile
    calibrator = SuspensionCalibrator(sampling_rate_hz=50.0)
    speed_vals = pd.to_numeric(df['speed_kmh'], errors='coerce').fillna(40.0).values if 'speed_kmh' in df.columns else None
    calib_res = calibrator.process_telemetry_stream(accel_z, speed_vals)
    z_road_mm = calib_res["deconvolved_profile_mm"]

    # Stack 6-channel physical array [6, N]
    raw_imu = np.stack([accel_z, dyn_accel, bandpass, jerk, gyro_y, z_road_mm], axis=0)

    # Normalize per-channel with robust z-score
    for c in range(6):
        m = np.median(raw_imu[c])
        s = np.std(raw_imu[c]) + 1e-4
        raw_imu[c] = np.clip((raw_imu[c] - m) / s, -6.0, 6.0)

    # Visual Features: [has_cam, box_area_pct, est_area_norm, est_depth_norm, asphalt_mass_norm, z_score]
    has_cam = (df['image_evidence'].notnull() & (df['image_evidence'] != 'None')).astype(float).values
    box_area = np.clip(pd.to_numeric(df['box_area_pct'], errors='coerce').fillna(0.0).values / 50.0, 0.0, 2.0)
    est_area = np.clip(pd.to_numeric(df['est_area_cm2'], errors='coerce').fillna(0.0).values / 15000.0, 0.0, 2.0)
    est_depth = np.clip(pd.to_numeric(df['est_depth_cm'], errors='coerce').fillna(0.0).values / 10.0, 0.0, 2.0)
    mass_norm = np.clip(pd.to_numeric(df['asphalt_mass_kg'], errors='coerce').fillna(0.0).values / 50.0, 0.0, 2.0)
    z_score = np.clip(pd.to_numeric(df['z_score'], errors='coerce').fillna(0.0).values / 5.0, 0.0, 2.0)

    vis_matrix = np.stack([has_cam, box_area, est_area, est_depth, mass_norm, z_score], axis=1)

    labels = pd.to_numeric(df['Label'], errors='coerce').fillna(0).astype(int).values
    sev_map = {'Low': 0, 'Moderate': 1, 'Severe': 2, 'None': 0}
    severities = df['Severity'].map(lambda x: sev_map.get(str(x), 0)).fillna(0).astype(int).values
    depths = pd.to_numeric(df['est_depth_cm'], errors='coerce').fillna(0.0).values

    half_w = window_size // 2
    n_samples = len(df)
    imu_windows = []
    selected_vis = []
    selected_labels = []
    selected_sevs = []
    selected_depths = []

    for i in range(half_w, n_samples - half_w, 2):
        window = raw_imu[:, i - half_w : i + half_w]
        imu_windows.append(window)
        selected_vis.append(vis_matrix[i])
        selected_labels.append(labels[i])
        selected_sevs.append(severities[i])
        selected_depths.append(depths[i])

    imu_windows = np.array(imu_windows, dtype=np.float32)
    selected_vis = np.array(selected_vis, dtype=np.float32)
    selected_labels = np.array(selected_labels, dtype=np.float32)
    selected_sevs = np.array(selected_sevs, dtype=np.int64)
    selected_depths = np.array(selected_depths, dtype=np.float32)

    print(f"  Extracted {len(selected_labels)} physics-aligned multi-modal samples.")
    p_count = int(np.sum(selected_labels))
    n_count = len(selected_labels) - p_count
    print(f"  Potholes: {p_count} ({p_count/len(selected_labels)*100:.1f}%), Normal: {n_count} ({n_count/len(selected_labels)*100:.1f}%)")

    return imu_windows, selected_vis, selected_labels, selected_sevs, selected_depths


# ====================================================================
# 6. TRAINING & EVALUATION LOOP
# ====================================================================

def train_cross_attention_model(
    data_path,
    epochs=12,
    batch_size=64,
    lr=1e-3,
    save_path="deep_cross_fusion.pt"
):
    print("=" * 70)
    print("ROADSENSE — Upgraded Multi-Modal Fusion Transformer (v2 Focal Loss)")
    print("=" * 70)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path, low_memory=False)
    imu_win, vis_feat, labels, sevs, depths = build_multimodal_windows(df, window_size=32)

    indices = np.arange(len(labels))
    train_idx, val_idx = train_test_split(indices, test_size=0.20, random_state=42, stratify=labels)

    train_ds = RoadSenseMultiModalDataset(imu_win[train_idx], vis_feat[train_idx], labels[train_idx], sevs[train_idx], depths[train_idx])
    val_ds = RoadSenseMultiModalDataset(imu_win[val_idx], vis_feat[val_idx], labels[val_idx], sevs[val_idx], depths[val_idx])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[TRAIN] Hardware Device: {device}")

    model = DeepCrossAttentionFusionNet(
        imu_channels=6,
        vis_features=6,
        d_model=64,
        nhead=4,
        dim_feedforward=128,
        seq_len=32,
        dropout=0.15
    ).to(device)

    # Binary Focal Loss with gamma=2.0 and alpha=0.65
    criterion_hazard = BinaryFocalLossWithLogits(alpha=0.65, gamma=2.0)
    criterion_sev = nn.CrossEntropyLoss()
    criterion_depth = nn.SmoothL1Loss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_f1 = 0.0
    best_auc = 0.0
    history = []

    print(f"\n[TRAIN] Starting optimization ({epochs} epochs, {len(train_loader)} batches/epoch)...")
    t0_all = time.time()

    for epoch in range(1, epochs + 1):
        t_ep_start = time.time()
        model.train()
        total_loss = 0.0

        for imu_b, vis_b, y_b, sev_b, depth_b in train_loader:
            imu_b = imu_b.to(device)
            vis_b = vis_b.to(device)
            y_b = y_b.to(device)
            sev_b = sev_b.to(device)
            depth_b = depth_b.to(device)

            optimizer.zero_grad()
            pred_logit, pred_sev, pred_depth = model(imu_b, vis_b)

            loss_haz = criterion_hazard(pred_logit, y_b)
            loss_sev = criterion_sev(pred_sev, sev_b)
            loss_dep = criterion_depth(pred_depth, depth_b)

            loss = loss_haz + (0.5 * loss_sev) + (0.25 * loss_dep)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        val_probs = []
        val_depth_errors = []

        with torch.no_grad():
            for imu_b, vis_b, y_b, _, depth_b in val_loader:
                imu_b = imu_b.to(device)
                vis_b = vis_b.to(device)
                pred_logit, _, pred_depth = model(imu_b, vis_b)
                probs = torch.sigmoid(pred_logit).cpu().numpy()
                preds = (probs >= 0.50).astype(int)

                val_probs.extend(probs)
                val_preds.extend(preds)
                val_targets.extend(y_b.numpy())
                val_depth_errors.extend(np.abs(pred_depth.cpu().numpy() - depth_b.numpy()))

        val_auc = float(roc_auc_score(val_targets, val_probs))
        val_f1 = float(f1_score(val_targets, val_preds, zero_division=0))
        val_prec = float(precision_score(val_targets, val_preds, zero_division=0))
        val_rec = float(recall_score(val_targets, val_preds, zero_division=0))
        mae_depth = float(np.mean(val_depth_errors))

        ep_time = round(time.time() - t_ep_start, 1)
        history.append({
            'epoch': epoch,
            'train_loss': round(avg_train_loss, 4),
            'val_auc': round(val_auc, 4),
            'val_f1': round(val_f1, 4),
            'val_precision': round(val_prec, 4),
            'val_recall': round(val_rec, 4),
            'mae_depth_cm': round(mae_depth, 3)
        })

        print(f"Epoch {epoch:02d}/{epochs:02d} [{ep_time}s] — Loss: {avg_train_loss:.4f} | AUC: {val_auc:.4f} | F1: {val_f1:.4f} | Prec: {val_prec:.4f} | Rec: {val_rec:.4f} | Depth MAE: {mae_depth:.2f}cm")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_auc = val_auc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'best_auc': val_auc,
                'config': {
                    'imu_channels': 6,
                    'vis_features': 6,
                    'd_model': 64,
                    'nhead': 4,
                    'seq_len': 32
                }
            }, save_path)

    total_time = round(time.time() - t0_all, 1)
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE & EVALUATION SCORECARD")
    print("=" * 70)
    print(f"Total Optimization Time: {total_time}s across {epochs} epochs")
    print(f"Peak Validation F1-Score: {best_f1*100:.2f}% | Peak ROC-AUC: {best_auc:.4f}")
    print(f"Trained Multi-Modal Model saved to: {save_path}")

    # Export ONNX Model
    export_onnx(model, save_path)

    # Save metrics JSON
    metrics_path = os.path.join(os.path.dirname(save_path), "fusion_transformer_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            'peak_val_f1': round(best_f1, 4),
            'final_val_auc': round(best_auc, 4),
            'training_time_sec': total_time,
            'epochs_history': history
        }, f, indent=2)
    print(f"Saved evaluation metrics to: {metrics_path}")

    return model, history


def export_onnx(model, weights_path):
    """Exports trained PyTorch model to ONNX runtime."""
    onnx_path = weights_path.replace(".pt", ".onnx")
    print(f"\n[EXPORT] Exporting to ONNX: {onnx_path}...")
    model.eval()
    dummy_imu = torch.randn(1, 6, 32)
    dummy_vis = torch.randn(1, 6)

    try:
        torch.onnx.export(
            model,
            (dummy_imu, dummy_vis),
            onnx_path,
            input_names=["imu_window", "vis_features"],
            output_names=["hazard_logit", "severity_logits", "depth_pred"],
            dynamic_axes={
                "imu_window": {0: "batch_size"},
                "vis_features": {0: "batch_size"},
                "hazard_logit": {0: "batch_size"},
                "severity_logits": {0: "batch_size"},
                "depth_pred": {0: "batch_size"}
            },
            opset_version=14
        )
        print(f"[SUCCESS] ONNX model saved to: {onnx_path} ({os.path.getsize(onnx_path)/1024:.1f} KB)")
    except Exception as e:
        print(f"[WARNING] ONNX export encountered error: {e}")


def load_cross_attention_model(weights_path=None, device=None):
    """Loads a pre-trained DeepCrossAttentionFusionNet from disk."""
    if weights_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        weights_path = os.path.join(base_dir, 'ml_pipeline', 'deep_cross_fusion.pt')

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found at: {weights_path}")

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(weights_path, map_location=device)
    cfg = checkpoint.get('config', {})

    model = DeepCrossAttentionFusionNet(
        imu_channels=cfg.get('imu_channels', 6),
        vis_features=cfg.get('vis_features', 6),
        d_model=cfg.get('d_model', 64),
        nhead=cfg.get('nhead', 4),
        seq_len=cfg.get('seq_len', 32)
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, device


def predict_cross_attention(model, imu_window, vis_features, device=None):
    """
    Runs multi-modal inference on a single sample or batch.
    """
    if device is None:
        device = next(model.parameters()).device

    if isinstance(imu_window, np.ndarray):
        imu_t = torch.tensor(imu_window, dtype=torch.float32)
    else:
        imu_t = imu_window.clone()

    if isinstance(vis_features, np.ndarray):
        vis_t = torch.tensor(vis_features, dtype=torch.float32)
    else:
        vis_t = vis_features.clone()

    if imu_t.dim() == 2:
        imu_t = imu_t.unsqueeze(0)
    if vis_t.dim() == 1:
        vis_t = vis_t.unsqueeze(0)

    imu_t = imu_t.to(device)
    vis_t = vis_t.to(device)

    model.eval()
    with torch.no_grad():
        h_logit, sev_logits, depth_pred, attn_w = model(imu_t, vis_t, return_attention=True)
        prob = torch.sigmoid(h_logit).cpu().numpy()
        sev_probs = F.softmax(sev_logits, dim=-1).cpu().numpy()
        depths = depth_pred.cpu().numpy()
        attn = attn_w.cpu().numpy()

    sev_labels = ['Minor / Low', 'Moderate', 'Severe']
    sev_idx = int(np.argmax(sev_probs[0]))

    return {
        'is_defect': bool(prob[0] >= 0.50),
        'confidence': float(prob[0]),
        'severity': sev_labels[sev_idx],
        'severity_probs': [float(p) for p in sev_probs[0]],
        'predicted_depth_cm': float(depths[0]),
        'attention_weights': attn[0].tolist()
    }


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    master_csv = os.path.join(base_dir, 'cv_pipeline', 'master_training_data.csv')
    weights_out = os.path.join(base_dir, 'ml_pipeline', 'deep_cross_fusion.pt')

    train_cross_attention_model(
        data_path=master_csv,
        epochs=12,
        batch_size=64,
        lr=1e-3,
        save_path=weights_out
    )
