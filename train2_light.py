import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from dataset import FermentationData

# ---------- 参数 ----------
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="Data5")
parser.add_argument("--batch_size", default=256, type=int)
parser.add_argument("--lr", default=0.02, type=float)
parser.add_argument("--hidden_dim", default=49, type=int)
parser.add_argument("--num_layers", default=1, type=int)
parser.add_argument("--num_epochs", default=50, type=int)
parser.add_argument("--no_cuda", action="store_true")
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
print("👉 使用设备：", DEVICE)

# ---------- 数据 ----------
train_ds = FermentationData(work_dir=args.dataset, train_mode=True)
test_ds = FermentationData(work_dir=args.dataset, train_mode=False)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

# ---------- 模型 ----------
class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h=None):
        out, h = self.lstm(x, h)
        out = self.fc(out[:, -1, :])
        return out, h

model = LSTMPredictor(
    input_dim=train_ds.get_num_features(),
    hidden_dim=args.hidden_dim,
    output_dim=1,
    n_layers=args.num_layers
).to(DEVICE)

opt = torch.optim.SGD(model.parameters(), lr=args.lr)
loss_fn = nn.MSELoss()

# ---------- 训练 ----------
best_rmse = 1e9
for epoch in range(1, args.num_epochs + 1):
    model.train()
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        pred, _ = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()

    model.eval()
    errs = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred, _ = model(x)
            errs.append(torch.sqrt(loss_fn(pred, y)).item())
    rmse = np.mean(errs)
    print(f"Epoch {epoch:03d} | RMSE: {rmse:.4f}")
    if rmse < best_rmse:
        best_rmse = rmse
        torch.save(model.state_dict(), "logs/weights_best.tar")
        print("  ↳ 保存最佳权重")

print("🎉 训练完成，最佳 RMSE:", best_rmse)