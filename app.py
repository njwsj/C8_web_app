'''
Description: Flask 后端服务 —— 接收上传的 DCS Excel 文件，使用训练好的 T3Time-Bi（路线A）
             模型预测 C8 选择性：
               • 历史回测：对上传区间做滚动一步预测
               • 未来预测：自回归方式向后推算任意天数

依赖安装:
  pip install flask pandas scipy scikit-learn torch openpyxl joblib

运行方式:
  cd code/web_app
  python app.py
  然后访问 http://localhost:5001
'''
import io
import os
import pickle

import joblib
import numpy as np
import pandas as pd
import pymysql
import torch
from flask import Flask, jsonify, render_template, request
from scipy.interpolate import CubicSpline
from torch import nn

# ─── 路径配置 ────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
ROOT_C8   = os.path.join(MODEL_DIR, 'C8选择性.xlsx')

# ─── 模型超参数（必须与 train_save.py 完全一致）────────────
DCS_COLS     = ['碳四进料流量', '调节剂进料流量', '入口温度', '热水流量', '小热水进出口温差', '入口温度SV']
WINDOW_HOURS = 6
SEQ_LENGTH   = 20
INPUT_SIZE   = 13   # N_VARS：6均值 + 6标准差 + 1 C8_spline
C8_IDX       = 12   # C8 特征在 feature_cols 中的索引（最后一列）
EMBED_DIM    = 16
N_HEADS      = 2
N_ENC_LAYERS = 2
N_DEC_LAYERS = 1

app = Flask(__name__)

# ─── MySQL 配置 ──────────────────────────────────────────────────
DB_CONFIG = {
    'host':    'localhost',
    'port':    3306,
    'user':    'root',
    'password': '123456',
    'charset': 'utf8mb4',
}
DB_NAME = 'c8_prediction'


def get_db():
    conn = pymysql.connect(**DB_CONFIG, database=DB_NAME)
    return conn


def init_db():
    conn = pymysql.connect(**DB_CONFIG)
    with conn.cursor() as cur:
        cur.execute(f'CREATE DATABASE IF NOT EXISTS `{DB_NAME}` CHARACTER SET utf8mb4')
        conn.select_db(DB_NAME)
        cur.execute('''
            CREATE TABLE IF NOT EXISTS prediction_history (
                id          INT AUTO_INCREMENT PRIMARY KEY,
                created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
                filename    VARCHAR(255),
                future_days INT,
                trend       VARCHAR(16),
                slope       FLOAT,
                hist_count  INT,
                fut_count   INT,
                hist_mean   FLOAT,
                hist_std    FLOAT,
                hist_min    FLOAT,
                hist_max    FLOAT,
                hist_ts     JSON,
                hist_preds  JSON,
                fut_ts      JSON,
                fut_preds   JSON,
                advice      JSON
            )
        ''')
    conn.commit()
    conn.close()


try:
    init_db()
except Exception as _e:
    print(f'[警告] 数据库初始化失败: {_e}')

# ─── 全局缓存（进程启动时加载一次，避免每次请求重复 IO）────────
_ARTIFACTS = None  # (model, scaler, close_max, close_min)
_C8_SPLINE  = None  # (cs, date0, c8_start, c8_end)


# ════════════════════════════════════════════════════════════════
# T3Time-Bi 模型定义（必须与 train_save.py 完全一致）
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
        # x: (B, N, L) — N个变量，L个历史时间步
        x_norm = self.revin.normalize(x)
        z_t = self.time_enc(x_norm)
        z_f = self.freq_enc(x_norm)
        z_g = self.gating(z_t, z_f, 1.0 / 96.0)
        z_d = self.decoder(z_g)
        out = self.readout(z_d)
        return self.revin.denormalize(out, self.c8_idx)


# ════════════════════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════════════════════
def load_artifacts():
    global _ARTIFACTS
    if _ARTIFACTS is not None:
        return _ARTIFACTS

    paths = {
        'model':  os.path.join(MODEL_DIR, 't3time_c.pth'),
        'scaler': os.path.join(MODEL_DIR, 'scaler.pkl'),
        'range':  os.path.join(MODEL_DIR, 'close_range.pkl'),
    }
    for p in paths.values():
        if not os.path.exists(p):
            return None, None, None, None

    model = T3TimeBi(
        n_vars  = INPUT_SIZE,
        in_len  = SEQ_LENGTH,
        c8_idx  = C8_IDX,
        C       = EMBED_DIM,
        n_heads = N_HEADS,
        n_enc   = N_ENC_LAYERS,
        n_dec   = N_DEC_LAYERS,
        dropout = 0.0,
    )
    ckpt = torch.load(paths['model'], map_location='cpu')
    model.load_state_dict(ckpt['state_dict'] if isinstance(ckpt, dict) else ckpt)
    model.eval()

    scaler = joblib.load(paths['scaler'])

    with open(paths['range'], 'rb') as f:
        close_max, close_min = pickle.load(f)

    _ARTIFACTS = (model, scaler, float(close_max), float(close_min))
    return _ARTIFACTS


def build_c8_spline():
    global _C8_SPLINE
    if _C8_SPLINE is not None:
        return _C8_SPLINE

    c8_df = pd.read_excel(ROOT_C8)
    c8_df.columns = c8_df.columns.str.strip()
    c8_df['date'] = pd.to_datetime(c8_df['检 验 日 期'].astype(str), format='%Y.%m.%d')
    c8_df = c8_df[['date', 'C8']].dropna()
    c8_daily = c8_df.groupby('date')['C8'].mean().reset_index().sort_values('date')
    date0 = c8_daily['date'].iloc[0]
    c8_daily['day_num'] = (c8_daily['date'] - date0).dt.total_seconds() / 86400.0
    cs = CubicSpline(c8_daily['day_num'].values, c8_daily['C8'].values, bc_type='natural')

    _C8_SPLINE = (cs, date0, c8_daily['date'].iloc[0], c8_daily['date'].iloc[-1])
    return _C8_SPLINE


def preprocess_dcs(file_bytes, cs, date0, c8_start, c8_end, scaler):
    """
    处理 DCS Excel → 返回 (scaled_array, windows_timestamps, error_msg)
    scaled_array shape: (N, 13)
    """
    dcs_df = pd.read_excel(io.BytesIO(file_bytes))

    try:
        dcs_df['datetime'] = pd.to_datetime(
            dcs_df['Date'].astype(str) + ' ' + dcs_df['Time'].astype(str)
        )
    except Exception:
        dcs_df['datetime'] = pd.to_datetime(dcs_df['Date'])

    dcs_df['window'] = dcs_df['datetime'].dt.floor(f'{WINDOW_HOURS}h')

    missing = [c for c in DCS_COLS if c not in dcs_df.columns]
    if missing:
        return None, None, f'缺少列：{missing}'

    dcs_mean = dcs_df.groupby('window')[DCS_COLS].mean()
    dcs_std  = dcs_df.groupby('window')[DCS_COLS].std().fillna(0).add_suffix('_std')
    dcs_win  = pd.concat([dcs_mean, dcs_std], axis=1).reset_index()
    dcs_win.columns.name = None

    dcs_win['midpoint'] = dcs_win['window'] + pd.Timedelta(hours=WINDOW_HOURS / 2)

    buf_start = c8_start - pd.Timedelta(days=1)
    buf_end   = c8_end   + pd.Timedelta(days=1)
    mask = (dcs_win['midpoint'] >= buf_start) & (dcs_win['midpoint'] <= buf_end)
    dcs_win = dcs_win[mask].copy()

    if len(dcs_win) < SEQ_LENGTH + 1:
        return None, None, (
            f'有效时间窗口数不足（需要 ≥ {SEQ_LENGTH + 1} 个，'
            f'当前仅 {len(dcs_win)} 个）。\n'
            f'请上传时间范围在 {c8_start.date()} ~ {c8_end.date()} 内、'
            f'跨度 ≥ {(SEQ_LENGTH + 1) * WINDOW_HOURS} 小时的 DCS 数据。'
        )

    dcs_win['day_num']   = (dcs_win['midpoint'] - date0).dt.total_seconds() / 86400.0
    dcs_win['C8_spline'] = np.clip(cs(dcs_win['day_num'].values), 0, 100)

    std_cols     = [c + '_std' for c in DCS_COLS]
    feature_cols = DCS_COLS + std_cols + ['C8_spline']
    merged     = dcs_win[feature_cols].dropna().reset_index(drop=True)
    windows_ts = dcs_win['window'].reset_index(drop=True)

    scaled = scaler.transform(merged)
    return scaled, windows_ts, None


def rolling_predict(model, scaled, windows_ts, close_max, close_min):
    """历史回测：在上传区间内做滚动一步预测。"""
    x_list, ts_list = [], []
    for i in range(len(scaled) - SEQ_LENGTH):
        x_list.append(scaled[i:i + SEQ_LENGTH, :])          # (L, N)
        ts_list.append(windows_ts.iloc[i + SEQ_LENGTH])

    # T3Time 输入格式: (B, N, L) — 转置最后两个维度
    x_np     = np.array(x_list, dtype=np.float32)            # (B, L, N)
    x_tensor = torch.from_numpy(x_np.transpose(0, 2, 1))     # (B, N, L)
    with torch.no_grad():
        preds_norm = model(x_tensor).numpy().flatten()

    preds = preds_norm * (close_max - close_min) + close_min
    return [str(t) for t in ts_list], preds.tolist()


def future_predict(model, scaled, windows_ts, close_max, close_min, future_steps):
    """
    自回归未来预测：
      - 以上传数据的最后 SEQ_LENGTH 个窗口为初始上下文
      - 每步预测下一个窗口的 C8，并将其作为特征滚入下一步
      - DCS 特征保持上传数据最后一个窗口的值（无未来 DCS 时的合理假设）
    """
    context = scaled[-SEQ_LENGTH:].copy()     # (L, N) = (20, 13)
    last_dcs_scaled = scaled[-1, :12].copy()  # 前12列为 DCS 特征

    last_ts  = windows_ts.iloc[-1]
    interval = pd.Timedelta(hours=WINDOW_HOURS)

    future_ts   = []
    future_vals = []

    for step in range(future_steps):
        # T3Time 输入格式: (1, N, L)
        x = torch.from_numpy(
            context[np.newaxis].transpose(0, 2, 1).astype(np.float32)
        )
        with torch.no_grad():
            pred_norm = float(model(x).item())

        pred_val = pred_norm * (close_max - close_min) + close_min
        future_ts.append(str(last_ts + interval * (step + 1)))
        future_vals.append(round(pred_val, 4))

        # 构造下一行：DCS 复用最后已知值，C8 用本步预测值
        new_row = np.append(last_dcs_scaled, pred_norm)   # (13,)
        context = np.vstack([context[1:], new_row])        # 滑动窗口

    return future_ts, future_vals


# ════════════════════════════════════════════════════════════════
# 路由
# ════════════════════════════════════════════════════════════════
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/status')
def status():
    ready = all(
        os.path.exists(os.path.join(MODEL_DIR, f))
        for f in ('t3time_c.pth', 'scaler.pkl', 'close_range.pkl')
    )
    return jsonify({'model_ready': ready})


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    POST /api/predict
    Form fields:
      file         : DCS Excel 文件（必填）
      future_days  : 向后预测天数，0 表示仅回测（可选，默认 0）

    返回 JSON:
    {
      "history": { "timestamps": [...], "predictions": [...], "count": N },
      "future":  { "timestamps": [...], "predictions": [...], "count": M, "note": "..." },
      "stats":   { "mean", "std", "min", "max" }
    }
    """
    if 'file' not in request.files:
        return jsonify({'error': '请选择要上传的 DCS Excel 文件'}), 400

    f = request.files['file']
    if f.filename == '':
        return jsonify({'error': '文件名为空'}), 400

    try:
        future_days = int(request.form.get('future_days', 0))
    except ValueError:
        future_days = 0
    future_days  = max(0, min(future_days, 365))
    future_steps = future_days * (24 // WINDOW_HOURS)

    model, scaler, close_max, close_min = load_artifacts()
    if model is None:
        return jsonify({'error': '模型文件不存在，请先运行 train_save.py'}), 503

    file_bytes = f.read()

    try:
        cs, date0, c8_start, c8_end = build_c8_spline()
    except Exception as e:
        return jsonify({'error': f'无法读取 C8 原始数据: {e}'}), 500

    try:
        scaled, windows_ts, err = preprocess_dcs(file_bytes, cs, date0, c8_start, c8_end, scaler)
    except Exception as e:
        return jsonify({'error': f'数据解析失败: {e}'}), 400

    if err:
        return jsonify({'error': err}), 400

    # ── 历史回测 ──
    hist_ts, hist_preds = rolling_predict(model, scaled, windows_ts, close_max, close_min)

    result = {
        'history': {
            'timestamps':  hist_ts,
            'predictions': hist_preds,
            'count':       len(hist_preds),
        },
        'stats': {
            'mean': round(float(np.mean(hist_preds)), 2),
            'std':  round(float(np.std(hist_preds)),  2),
            'min':  round(float(np.min(hist_preds)),  2),
            'max':  round(float(np.max(hist_preds)),  2),
        }
    }

    # ── 未来预测 ──
    if future_steps > 0:
        fut_ts, fut_preds = future_predict(
            model, scaled, windows_ts, close_max, close_min, future_steps
        )
        result['future'] = {
            'timestamps':  fut_ts,
            'predictions': fut_preds,
            'count':       len(fut_preds),
            'note':        f'DCS 特征取上传数据末尾窗口均值，每步预测跨度 {WINDOW_HOURS} 小时'
        }

    # ── 趋势分析与调节建议 ──
    all_preds = hist_preds + (fut_preds if future_steps > 0 else [])
    if len(all_preds) >= 4:
        recent = all_preds[-4:]
        slope = (recent[-1] - recent[0]) / 3
        if slope < -0.05:
            trend = 'down'
            advice = [
                '适当降低入口温度（过高温度会导致副反应增加）',
                '检查调节剂进料流量是否偏低（调节剂不足会降低选择性）',
                '检查碳四进料流量是否过大（空速过高导致接触时间不足）',
                '检查热水流量是否异常（影响反应温度均匀性）',
            ]
        elif slope > 0.05:
            trend = 'up'
            advice = [
                '当前参数组合较优，可维持现有操作条件',
                '可小幅提高调节剂进料流量以进一步强化选择性',
                '记录当前参数作为最优操作点参考',
            ]
        else:
            trend = 'stable'
            advice = ['预测曲线平稳，当前操作条件良好，建议维持现有参数。']
        result['trend'] = {'direction': trend, 'slope': round(slope, 4), 'advice': advice}

    # ── 保存到数据库 ──
    try:
        import json
        tr = result.get('trend', {})
        fu = result.get('future', {})
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute('''
                INSERT INTO prediction_history
                  (filename, future_days, trend, slope, hist_count, fut_count,
                   hist_mean, hist_std, hist_min, hist_max,
                   hist_ts, hist_preds, fut_ts, fut_preds, advice)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ''', (
                f.filename, future_days,
                tr.get('direction'), tr.get('slope'),
                result['history']['count'], fu.get('count', 0),
                result['stats']['mean'], result['stats']['std'],
                result['stats']['min'],  result['stats']['max'],
                json.dumps(hist_ts, ensure_ascii=False),
                json.dumps(hist_preds),
                json.dumps(fu.get('timestamps', []), ensure_ascii=False),
                json.dumps(fu.get('predictions', [])),
                json.dumps(tr.get('advice', []), ensure_ascii=False),
            ))
        conn.commit()
        conn.close()
    except Exception as _db_err:
        print(f'[警告] 保存预测历史失败: {_db_err}')

    return jsonify(result)


@app.route('/api/history')
def history():
    """返回最近 50 条预测历史（不含原始时间序列数据）"""
    try:
        conn = get_db()
        with conn.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute('''
                SELECT id, created_at, filename, future_days, trend, slope,
                       hist_count, fut_count, hist_mean, hist_std, hist_min, hist_max, advice
                FROM prediction_history
                ORDER BY created_at DESC
                LIMIT 50
            ''')
            rows = cur.fetchall()
        conn.close()
        for r in rows:
            r['created_at'] = str(r['created_at'])
            if isinstance(r['advice'], str):
                import json
                r['advice'] = json.loads(r['advice'])
        return jsonify({'records': rows})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/history/<int:record_id>')
def history_detail(record_id):
    """返回某条历史记录的完整时间序列数据"""
    try:
        import json
        conn = get_db()
        with conn.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute('SELECT * FROM prediction_history WHERE id=%s', (record_id,))
            row = cur.fetchone()
        conn.close()
        if not row:
            return jsonify({'error': '记录不存在'}), 404
        row['created_at'] = str(row['created_at'])
        for col in ('hist_ts', 'hist_preds', 'fut_ts', 'fut_preds', 'advice'):
            if isinstance(row[col], str):
                row[col] = json.loads(row[col])
        return jsonify(row)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 50)
    print(" C8 选择性预测服务（T3Time-Bi 路线A）")
    print("=" * 50)
    if not os.path.exists(os.path.join(MODEL_DIR, 't3time_c.pth')):
        print("[警告] 模型文件不存在！请先运行:\n  python train_save.py")
    else:
        print("[OK] 模型文件就绪")
    print("\n访问地址: http://localhost:5001")
    print("[提示] 生产部署请使用: gunicorn -w 1 -b 0.0.0.0:5001 --timeout 120 app:app\n")
    app.run(debug=False, host='0.0.0.0', port=5001)
