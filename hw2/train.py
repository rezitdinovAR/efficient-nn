import argparse
import json
import math
import os
import platform
import time

import torch
import torch.nn.functional as F

from data import TrainLoader, evaluate, load_cifar10
from model import build_model
from schedules import Schedule

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
DATA = os.path.join(HERE, "data")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


def flat_params(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--policy", default="1cycle", choices=["const", "step", "range", "1cycle", "triangular"])
    p.add_argument("--depth", type=int, default=56)
    p.add_argument("--bs", type=int, default=512)
    p.add_argument("--epochs", type=float, default=None)
    p.add_argument("--iters", type=int, default=None, help="если задано, --epochs игнорируется")
    p.add_argument("--lr", type=float, default=0.35, help="для const/step")
    p.add_argument("--lr-min", type=float, default=0.1)
    p.add_argument("--lr-max", type=float, default=3.0)
    p.add_argument("--pct-up", type=float, default=0.45)
    p.add_argument("--pct-down", type=float, default=0.45)
    p.add_argument("--final-div", type=float, default=100.0)
    p.add_argument("--mom", type=float, default=0.9)
    p.add_argument("--mom-max", type=float, default=None)
    p.add_argument("--mom-min", type=float, default=None)
    p.add_argument("--milestones", type=float, nargs="+", default=[0.625, 0.875])
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--nesterov", action="store_true")
    p.add_argument("--optimizer", default="sgd", choices=["sgd", "adam"])
    p.add_argument("--bn-maf", type=float, default=0.95, help="moving average fraction для BN, как в caffe")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--n-train", type=int, default=None)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--eval-every", type=int, default=None, help="итераций между проверками на тесте (по умолчанию эпоха)")
    p.add_argument("--hf-estimate", type=int, default=0, help="считать оценку lr по формуле (8) первые N итераций")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--amp", action="store_true")
    a = p.parse_args()

    torch.manual_seed(a.seed)
    dev = get_device()
    out_dir = os.path.join(RES, a.name)
    os.makedirs(out_dir, exist_ok=True)

    xtr, ytr, xte, yte = load_cifar10(DATA, dev, a.n_train, a.seed)
    loader = TrainLoader(xtr, ytr, a.bs, augment=not a.no_augment, seed=a.seed)
    ipe = loader.iters_per_epoch
    T = a.iters if a.iters is not None else int(round(a.epochs * ipe))
    eval_every = a.eval_every or ipe

    model = build_model(a.depth, bn_momentum=1.0 - a.bn_maf, dropout=a.dropout).to(dev)
    sched = Schedule(a.policy, T, lr=a.lr, lr_max=a.lr_max, lr_min=a.lr_min, pct_up=a.pct_up,
                     pct_down=a.pct_down, final_div=a.final_div, mom=a.mom, mom_max=a.mom_max,
                     mom_min=a.mom_min, milestones=tuple(a.milestones))
    lr0, m0 = sched(0)
    if a.optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr0, momentum=m0, weight_decay=a.wd, nesterov=a.nesterov)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr0, betas=(m0, 0.999), weight_decay=a.wd)

    meta = dict(vars(a), T=T, iters_per_epoch=ipe, n_train=len(xtr), device=str(dev),
                torch=torch.__version__, platform=platform.platform(),
                n_params=sum(q.numel() for q in model.parameters()))
    print(json.dumps(meta, indent=1))
    log = open(os.path.join(out_dir, "log.csv"), "w")
    log.write("iter,epoch,lr,mom,train_loss,train_acc,test_loss,test_acc,time_s\n")
    hf_log = None
    if a.hf_estimate:
        hf_log = open(os.path.join(out_dir, "hf_estimate.csv"), "w")
        hf_log.write("iter,lr,eps_star,eps_star_ema\n")

    it = iter(loader)
    run_loss, run_correct, run_n = 0.0, 0, 0
    snaps, ema = [], None
    best = 0.0
    sync(dev)
    t0 = time.time()
    model.train()
    for i in range(T):
        lr, mom = sched(i)
        for g in opt.param_groups:
            g["lr"] = lr
            if a.optimizer == "sgd":
                g["momentum"] = mom
            else:
                g["betas"] = (mom, 0.999)
        xb, yb = next(it)
        if a.amp:
            with torch.autocast(dev.type, dtype=torch.bfloat16):
                out = model(xb)
                loss = F.cross_entropy(out, yb)
        else:
            out = model(xb)
            loss = F.cross_entropy(out, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if hf_log and i < a.hf_estimate:
            # eps* = eps * sum|th1-th0| / sum|2*th1 - th0 - th2|, по трём последним весам
            snaps.append(flat_params(model).clone())
            if len(snaps) == 3:
                th0, th1, th2 = snaps
                eps = (lr * (th1 - th0).abs().sum() / (2 * th1 - th0 - th2).abs().sum()).item()
                ema = eps if ema is None else 0.1 * eps + 0.9 * ema
                hf_log.write(f"{i},{lr:.6g},{eps:.6g},{ema:.6g}\n")
                snaps.pop(0)

        run_loss += loss.item() * len(yb)
        run_correct += (out.argmax(1) == yb).sum().item()
        run_n += len(yb)
        if (i + 1) % eval_every == 0 or i + 1 == T:
            te_loss, te_acc = evaluate(model, xte, yte)
            sync(dev)
            el = time.time() - t0
            tr_loss, tr_acc = run_loss / run_n, run_correct / run_n
            run_loss, run_correct, run_n = 0.0, 0, 0
            best = max(best, te_acc)
            log.write(f"{i+1},{(i+1)/ipe:.4f},{lr:.6g},{mom:.4g},{tr_loss:.5f},{tr_acc:.5f},{te_loss:.5f},{te_acc:.5f},{el:.1f}\n")
            log.flush()
            print(f"[{a.name}] it {i+1}/{T} ep {(i+1)/ipe:6.2f} lr {lr:.4f} mom {mom:.3f} "
                  f"train {tr_loss:.3f}/{tr_acc*100:.2f}% test {te_loss:.3f}/{te_acc*100:.2f}% ({el/60:.1f} min)",
                  flush=True)
            if not math.isfinite(tr_loss):
                print("nan, stop")
                break
    log.close()
    if hf_log:
        hf_log.close()
    te_loss, te_acc = evaluate(model, xte, yte)
    meta.update(final_test_acc=te_acc, final_test_loss=te_loss, best_test_acc=best, wall_time_s=time.time() - t0)
    json.dump(meta, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    print(f"[{a.name}] DONE final test acc {te_acc*100:.2f}% (best {best*100:.2f}%) in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
