# 🚀 一键部署指南（小学生级）

> 你只需要做 4 件事：装软件 → 拿 Key → 跑脚本 → 看结果。

---

## ✅ 你需要先有的 3 个 Key

| Key | 在哪拿 | 必须吗 |
|---|---|---|
| **LLM API Key**（推荐 DeepSeek，便宜） | https://platform.deepseek.com/ | ✅ 必须 |
| **Binance Futures Testnet API Key + Secret** | https://testnet.binancefuture.com/ | ✅ 必须（不烧真钱） |
| **Dashboard Token** | 不用拿，脚本会自动生成 | 自动 |

> Telegram 机器人 / Square cookie / 代理 都是**可选**，先不管。

---

## 🐳 Step 1：装 Docker Desktop（5 分钟）

| 系统 | 下载 | 装完做什么 |
|---|---|---|
| Windows | https://www.docker.com/products/docker-desktop/ | **重启电脑** + 双击打开它 + 等托盘里鲸鱼图标变静止 |
| Mac (M 芯片或 Intel) | 同上，选对应版本 | 打开它 + 等鲸鱼图标变静止 |
| Linux | `curl -fsSL https://get.docker.com \| sh` | `sudo systemctl start docker` |

**判断装好的标准：** 打开终端敲 `docker info`，没报错就行。

---

## 📦 Step 2：下载代码

```bash
git clone https://github.com/yiling-ui/-.git altcoin-agent
cd altcoin-agent
git checkout plan-and-tasks-START-HERE
```

---

## 🎯 Step 3：一键部署

### 🐧 Linux / 🍎 macOS / WSL

```bash
chmod +x deploy.sh
./deploy.sh
```

### 🪟 Windows（PowerShell，**右键 → 以管理员身份运行**）

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\deploy.ps1
```

---

## 🔑 Step 4：脚本会要求你做什么

第一次运行时它会：

1. **检查 Docker** — 没装/没启会告诉你怎么修
2. **生成 `.env` 文件** — 自动写一个随机 `DASHBOARD_TOKEN`（请抄下来！）
3. **报错说缺 LLM Key** — 这时你要：
   ```bash
   nano .env       # Linux/Mac
   notepad .env    # Windows
   ```
   找到 `DEEPSEEK_API_KEY=` 把你的 key 贴进去保存。
4. **再跑一次脚本** — 这次它会构建镜像、启动、做健康检查。

第一次构建大约要 **3–8 分钟**（看网速），后面就是秒级。

---

## ✨ 成功之后

终端会输出：

```
========================================
  Deploy successful
========================================
  Dashboard:  http://localhost:8080/dashboard
  Token:      a3f9e8c1d2b4...
  Mode:       DRY_RUN=true  PAPER_TRADE=false
```

打开浏览器访问 **http://localhost:8080/dashboard**，输入 Token 就能看面板了。

---

## 🔧 中国大陆网络问题怎么办

### 方案 A：脚本自带（自动）

`deploy.sh` 会自动检测你的网络，若访问 google 超时但能访问 baidu，就会自动切到清华 pip 镜像。

### 方案 B：手动指定镜像（强力）

在 `.env` 里加：

```bash
APT_MIRROR=mirrors.tuna.tsinghua.edu.cn
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
```

然后 `./deploy.sh` 重跑。

### 方案 C：Docker Desktop 镜像加速

打开 Docker Desktop → Settings → Docker Engine，把这一段贴进 JSON：

```json
{
  "registry-mirrors": [
    "https://docker.m.daocloud.io",
    "https://dockerproxy.com",
    "https://mirror.baidubce.com"
  ]
}
```

点 **Apply & Restart**。

### 方案 D：还是连不上 LLM？

在 `.env` 里加代理：

```bash
PROXY_POOL=http://用户名:密码@代理ip:端口
PROXY_ROTATE=true
```

---

## 🆘 常见报错对照表

| 报错关键字 | 原因 | 解决 |
|---|---|---|
| `Cannot connect to the Docker daemon` | Docker 没启 | 打开 Docker Desktop 等鲸鱼变静止 |
| `port is already allocated` | 8080 端口被别的程序占用 | 改 `.env` 里 `HEALTHZ_PORT=8081` |
| `network timeout` / `dial tcp: i/o timeout` 在 build 阶段 | 网络问题 | 用上面的方案 B 或 C |
| `Invalid API-key` | LLM key 错了 | 检查 `.env` 里 key 有没有空格或换行 |
| `service failed to become healthy` | 启动后 90s 没响应 | 看 `docker compose logs --tail=200` |
| Windows 报 `cannot be loaded because running scripts is disabled` | PS 默认禁脚本 | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |

---

## 📋 日常维护命令

```bash
docker compose logs -f          # 实时看日志（Ctrl+C 退出）
docker compose ps               # 查看运行状态
docker compose restart          # 重启
docker compose down             # 停掉
docker compose pull             # 拉新镜像
./deploy.sh                     # 改完 .env 后再跑一次（幂等安全）
```

---

## 🛑 紧急停止交易

```bash
# 1. 改 .env 里 DRY_RUN=true
# 2. 重启
docker compose restart
```

或直接：

```bash
docker compose down             # 完全停
```

---

## 📊 接下来：开始记录 30 天 dry-run 数据

部署成功后参考根目录下 `PLAN_README.md` 的"📊 测试数据收集清单"按天记录就好。
