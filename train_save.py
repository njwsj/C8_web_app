'''
T3Time-Bi 路线A 训练脚本
  数据处理：路线A（剔除停产期伪影、分段构建序列、Scaler 仅训练集 fit）
  模型：T3Time 双模态（时域 + 频域）+ RevIN
  保存：model/t3time_c.pth、model/scaler.pkl、model/close_range.pkl

运行方式:
  cd code/web_app
  python train_save.py
'''
import os
import pickle
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.interpolate import CubicSpline
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ─── 路径 ────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
ROOT_C8   = r"/Users/zhanghongjia/Downloads/3.12处理数据/01_源数据/C8选择性.xlsx"
ROOT_DCS  = r"/Users/zhanghongjia/Downloads/3.12处理数据/01_源数据/dcs数据.xlsx"
os.makedirs(MODEL_DIR, exist_ok=True)

DCS_COLS     = ['碳四进料流量', '调节剂进料流量', '入口温度', '热水流量', '小热水进出口温差', '入口温度SV']
WINDOW_HOURS = 6
SEQ_LENGTH   = 20
EMBED_DIM    = 16
N_HEADS      = 2
N_ENC_LAYERS = 2
N_DEC_LAYERS = 1
DROPOUT_RATE = 0.3
EPOCHS       = 300
BATCH_SIZE   = 32
LR           = 0.001
WEIGHT_DECAY = 1e-4
PATIENCE     = 40

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# ════════════════════════════════════════════════════════════════
# Dataset
# ════════════════════════════════════════════════════════════════
class TimeSeriesDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x)  # (B, N, L)
        self.y = torch.from_numpy(y)  # (B, 1)

    def __getitem__(self, i): return self.x[i], self.y[i]
    def __len__(self): return len(self.x)


# ════════════════════════════════════════════════════════════════
# T3Time-Bi 模型组件
# ════════════════════════════════════════════════════════════════
class RevIN(nn.Module):
    def __init__(self, n_vars, eps=1e-5):
        super().__init__()
        self.eps   = eps
        self.gamma = nn.Parameter(torch.ones(n_vars))
        self.beta  = nn.Parameter(torch.zeros(n_vars))

    def normalize(self, x):
        self._mean = x.mean(dim=-1, keepdim=True)
        self._std  = x.std(dim=-1, keepdim=True).clamp(self.eps)
        return (x - self._mean) / self._std * self.gamma[None, :, None] + self.beta[None, :, None]

    def denormalize(self, y, c8_idx):
        y = (y - self.beta[c8_idx]) / self.gamma[c8_idx].clamp(self.eps)
        return y * self._std[:, c8_idx, :] + self._mean[:, c8_idx, :]


class TransformerBlock(nn.Module):
    def __init__(self, d, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.ff    = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d))
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h)
        x = x + self.drop(a)
        return x + self.drop(self.ff(self.norm2(x)))


class TimeBranch(nn.Module):
    def __init__(self, in_len, C, n_heads, n_layers, dropout):
        super().__init__()
        self.proj = nn.Linear(in_len, C)
        self.enc  = nn.Sequential(*[TransformerBlock(C, n_heads, C * 4, dropout) for _ in range(n_layers)])

    def forward(self, x): return self.enc(self.proj(x))


class FreqBranch(nn.Module):
    def __init__(self, in_len, C, n_heads, n_layers, dropout):
        super().__init__()
        self.proj = nn.Linear(in_len // 2 + 1, C)
        self.enc  = nn.Sequential(*[TransformerBlock(C, n_heads, C * 4, dropout) for _ in range(n_layers)])

    def forward(self, x): return self.enc(self.proj(torch.fft.rfft(x, dim=-1).abs()))


class HorizonGating(nn.Module):
    def __init__(self, C, dropout=0.1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(C + 1, C), nn.ReLU(), nn.Dropout(dropout), nn.Linear(C, C), nn.Sigmoid()
        )

    def forward(self, z_t, z_f, norm_horizon):
        B, N, C = z_t.shape
        g = self.fc(torch.cat([z_t.mean(dim=1), z_t.new_full((B, 1), norm_horizon)], dim=-1)).unsqueeze(1)
        return g * z_f + (1 - g) * z_t


class T3TimeBi(nn.Module):
    def __init__(self, n_vars, in_len, c8_idx, C=16, n_heads=2, n_enc=2, n_dec=1, dropout=0.1):
        super().__init__()
        self.c8_idx   = c8_idx
        self.revin    = RevIN(n_vars)
        self.time_enc = TimeBranch(in_len, C, n_heads, n_enc, dropout)
        self.freq_enc = FreqBranch(in_len, C, n_heads, n_enc, dropout)
        self.gating   = HorizonGating(C, dropout)
        self.decoder  = nn.Sequential(*[TransformerBlock(C, n_heads, C * 4, dropout) for _ in range(n_dec)])
        self.readout  = nn.Sequential(
            nn.Flatten(), nn.Linear(n_vars * C, C), nn.GELU(), nn.Dropout(dropout), nn.Linear(C, 1)
        )

    def forward(self, x):
        x_norm = self.revin.normalize(x)
        z_t = self.time_enc(x_norm)
        z_f = self.freq_enc(x_norm)
        z_g = self.gating(z_t, z_f, 1.0 / 96.0)
        z_d = self.decoder(z_g)
        out = self.readout(z_d)
        return self.revin.denormalize(out, self.c8_idx)


# ════════════════════════════════════════════════════════════════
# 数据加载（路线A）
# ════════════════════════════════════════════════════════════════
def getData():
    C8_THRESHOLD  = 50.0
    GAP_THRESHOLD = 7

    print("[Step 1] 读取C8，识别停产期...")
    c8_df = pd.read_excel(ROOT_C8)
    c8_df.columns = c8_df.columns.str.strip()
    c8_df['date'] = pd.to_datetime(c8_df['检 验 日 期'].astype(str), format='%Y.%m.%d')
    c8_df = c8_df[['date', 'C8']].dropna()
    c8_daily = c8_df.groupby('date')['C8'].mean().reset_index().sort_values('date').reset_index(drop=True)
    date0 = c8_daily['date'].iloc[0]
    c8_daily['day_num'] = (c8_daily['date'] - date0).dt.total_seconds() / 86400.0
    cs = CubicSpline(c8_daily['day_num'].values, c8_daily['C8'].values, bc_type='natural')

    c8_daily['gap_days'] = c8_daily['date'].diff().dt.days
    gap_periods = []
    for idx in c8_daily[c8_daily['gap_days'] > GAP_THRESHOLD].index:
        gs, ge = c8_daily.loc[idx - 1, 'date'], c8_daily.loc[idx, 'date']
        gap_periods.append((gs, ge))
        print(f"    停产期: {gs.date()} ~ {ge.date()} ({int(c8_daily.loc[idx, 'gap_days'])} 天)")
    print(f"    C8实测天数: {len(c8_daily)}")

    print(f"[Step 2] 读取DCS，按 {WINDOW_HOURS}h 分桶...")
    dcs_df = pd.read_excel(ROOT_DCS)
    try:
        dcs_df['datetime'] = pd.to_datetime(
            dcs_df['Date'].astype(str) + ' ' + dcs_df['Time'].astype(str))
    except Exception:
        dcs_df['datetime'] = pd.to_datetime(dcs_df['Date'])
    dcs_df['window'] = dcs_df['datetime'].dt.floor(f'{WINDOW_HOURS}h')
    dcs_mean = dcs_df.groupby('window')[DCS_COLS].mean()
    dcs_std  = dcs_df.groupby('window')[DCS_COLS].std().fillna(0).add_suffix('_std')
    dcs_win  = pd.concat([dcs_mean, dcs_std], axis=1).reset_index()
    dcs_win.columns.name = None

    print("[Step 3] 样条估算C8...")
    dcs_win['midpoint'] = dcs_win['window'] + pd.Timedelta(hours=WINDOW_HOURS / 2)
    c8_start = c8_daily['date'].iloc[0]  - pd.Timedelta(days=1)
    c8_end   = c8_daily['date'].iloc[-1] + pd.Timedelta(days=1)
    dcs_win  = dcs_win[(dcs_win['midpoint'] >= c8_start) & (dcs_win['midpoint'] <= c8_end)].copy()
    dcs_win['day_num']   = (dcs_win['midpoint'] - date0).dt.total_seconds() / 86400.0
    dcs_win['C8_spline'] = np.clip(cs(dcs_win['day_num'].values), 0, 100)

    print("[Step 4] 路线A过滤：剔除停产期 + 开机爬坡...")
    n_raw = len(dcs_win)

    def _in_gap(m):
        return any(gs < m < ge for gs, ge in gap_periods)

    dcs_win = dcs_win[~dcs_win['midpoint'].map(_in_gap)].copy()
    dcs_win = dcs_win[dcs_win['C8_spline'] >= C8_THRESHOLD].copy()
    print(f"    有效窗口: {len(dcs_win)} / {n_raw}（剔除 {n_raw - len(dcs_win)} 个）")

    std_cols     = [c + '_std' for c in DCS_COLS]
    feature_cols = DCS_COLS + std_cols + ['C8_spline']
    n_vars = len(feature_cols)   # 13
    c8_idx = n_vars - 1

    merged = dcs_win[['window'] + feature_cols].dropna().sort_values('window').reset_index(drop=True)
    time_diff_h = merged['window'].diff().dt.total_seconds() / 3600
    merged['seg_id'] = (time_diff_h > WINDOW_HOURS * 2).cumsum().fillna(0).astype(int)

    print("    过滤后连续生产段：")
    for sid, grp in merged.groupby('seg_id'):
        print(f"      段{sid}: {grp['window'].iloc[0].date()} ~ {grp['window'].iloc[-1].date()}, {len(grp)} 个窗口")

    n_total    = len(merged)
    split_t    = int(0.8 * n_total)
    train_mask = merged.index < split_t

    scaler = MinMaxScaler()
    scaler.fit(merged.loc[train_mask, feature_cols].values)
    all_scaled = scaler.transform(merged[feature_cols].values).astype(np.float32)

    close_min = float(scaler.data_min_[c8_idx])
    close_max = float(scaler.data_max_[c8_idx])
    print(f"    C8 归一化基准: [{close_min:.2f}%, {close_max:.2f}%]")

    x_list, y_list, is_train_list = [], [], []
    for sid, seg_df in merged.groupby('seg_id'):
        idx_arr    = seg_df.index.values
        seg_scaled = all_scaled[idx_arr]
        seg_train  = train_mask[idx_arr]
        n = len(seg_scaled)
        if n <= SEQ_LENGTH:
            print(f"    段{sid}: 仅 {n} 个窗口，跳过")
            continue
        for i in range(n - SEQ_LENGTH):
            x_list.append(seg_scaled[i:i + SEQ_LENGTH].T)   # (N, L)
            y_list.append(seg_scaled[i + SEQ_LENGTH, c8_idx])
            is_train_list.append(bool(seg_train[i + SEQ_LENGTH]))

    x_arr    = np.array(x_list, dtype=np.float32)
    y_arr    = np.array(y_list, dtype=np.float32).reshape(-1, 1)
    is_train = np.array(is_train_list)

    trainx, trainy = x_arr[is_train],  y_arr[is_train]
    testx,  testy  = x_arr[~is_train], y_arr[~is_train]
    print(f"    训练: {len(trainy)} 条，测试: {len(testy)} 条")

    train_loader = DataLoader(TimeSeriesDataset(trainx, trainy), batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(TimeSeriesDataset(testx,  testy),  batch_size=BATCH_SIZE, shuffle=False)
    return close_max, close_min, train_loader, test_loader, n_vars, c8_idx, scaler


# ════════════════════════════════════════════════════════════════
# 训练
# ════════════════════════════════════════════════════════════════
def train_model(model, train_loader, test_loader):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)
    best_val, best_state, no_imp = float('inf'), None, 0

    for ep in range(EPOCHS):
        model.train()
        batch_l = []
        for x, y in tqdm(train_loader, leave=False, desc=f'epoch {ep + 1}/{EPOCHS}'):
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_l.append(loss.item())

        model.eval()
        val_l = []
        with torch.no_grad():
            for x, y in test_loader:
                val_l.append(criterion(model(x.to(device)), y.to(device)).item())
        vl = float(np.mean(val_l))
        scheduler.step(vl)

        if vl < best_val:
            best_val  = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_imp    = 0
        else:
            no_imp += 1

        if (ep + 1) % 20 == 0:
            print(f"  epoch {ep+1:3d} | train={np.mean(batch_l):.6f} | val={vl:.6f} | best={best_val:.6f}")

        if no_imp >= PATIENCE:
            print(f"\n  Early stop at epoch {ep + 1}, best val_loss={best_val:.6f}")
            break

    model.load_state_dict(best_state)
    print(f"\n  最佳 val_loss={best_val:.6f}")
    return model, best_val


# ════════════════════════════════════════════════════════════════
# 主程序：训练 → 保存
# ════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print(f"使用设备: {device}")
    close_max, close_min, train_loader, test_loader, n_vars, c8_idx, scaler = getData()

    model = T3TimeBi(
        n_vars  = n_vars,
        in_len  = SEQ_LENGTH,
        c8_idx  = c8_idx,
        C       = EMBED_DIM,
        n_heads = N_HEADS,
        n_enc   = N_ENC_LAYERS,
        n_dec   = N_DEC_LAYERS,
        dropout = DROPOUT_RATE,
    )
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    model, best_val = train_model(model, train_loader, test_loader)

    # ── 保存模型权重 ──
    model_path = os.path.join(MODEL_DIR, 't3time_c.pth')
    torch.save({'state_dict': model.state_dict(), 'best_val': best_val}, model_path)
    print(f"\n模型已保存: {model_path}")

    # ── 保存 Scaler ──
    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"Scaler 已保存: {scaler_path}")

    # ── 保存 C8 值域 ──
    range_path = os.path.join(MODEL_DIR, 'close_range.pkl')
    with open(range_path, 'wb') as f:
        pickle.dump((close_max, close_min), f)
    print(f"C8 值域已保存: {range_path}  [{close_min:.2f}%, {close_max:.2f}%]")

    print("\n所有文件已保存，可以启动 app.py 提供预测服务。")
