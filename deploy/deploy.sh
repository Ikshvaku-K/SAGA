#!/usr/bin/env bash
# Deploy SAGA (platform + mobile web app + report API)
# to a Linux server (Ubuntu/Debian, systemd, root SSH). All inference runs on the server.
#
#   AG_HOST=root@1.2.3.4 AG_SSH_PORT=22 deploy/deploy.sh check
#   ... push       code, model, metrics, and the images the server needs to rebuild the map
#   ... setup      venv (CUDA or CPU PyTorch, auto), Caddy (or cloudflared), firewall
#   ... rebuild    test-set eval, audit scores, gallery, and city map, computed ON the server
#   ... login      gate questions/answers (AG_GATE_QUESTIONS, AG_GATE_ANSWERS) + shared password; only hashes are sent
#   ... start      systemd services: API (127.0.0.1:8010), HTTPS (Caddy for $AG_DOMAIN, else tunnel)
#   ... url        print the public HTTPS address
#   ... status     service state + last log lines
#   ... voice      ElevenLabs voice: prompts for the API key (hidden); optional AG_VOICE_ID / AG_VOICE_MODEL
#
# Server layout (nothing outside these paths is touched):
#   $AG_DIR (default /opt/saga)   code, .venv, models/, data/, results/
#   /etc/saga/env                 AG_MODEL, AG_ADMIN_TOKEN, login hashes, optional AG_ELEVENLABS_* keys
#   /etc/systemd/system/saga-*.service
set -euo pipefail

HOST=${AG_HOST:?set AG_HOST=user@server}
PORT=${AG_SSH_PORT:-22}
KEY=${AG_SSH_KEY:-$HOME/.ssh/id_ed25519}
DIR=${AG_DIR:-/opt/saga}
RUN=${AG_RUN:-b2_8k}                       # which remote_runs/<run> model to deploy
DOMAIN=${AG_DOMAIN-sagatech.us}             # empty -> Cloudflare quick tunnel instead of Caddy
LOCAL=$(cd "$(dirname "$0")/.." && pwd)
SSH=(ssh -p "$PORT" -i "$KEY" -o IdentitiesOnly=yes -o ServerAliveInterval=30 "$HOST")
RSYNC=(rsync -az --stats -e "ssh -p $PORT -i $KEY -o IdentitiesOnly=yes")
TMP=${TMPDIR:-/tmp}

cmd=${1:-help}; shift || true

case "$cmd" in
check)
  "${SSH[@]}" "set +e
    . /etc/os-release; echo \"os: \$PRETTY_NAME\"; echo \"cpus: \$(nproc)\"; free -g | sed -n 2p; df -h / | tail -1
    command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader || echo 'gpu: none (CPU inference, ~1-2 s per photo)'
    command -v systemctl >/dev/null && echo 'systemd: yes' || echo 'systemd: MISSING'
    python3 --version; command -v uv cloudflared tailscale 2>/dev/null
    ss -ltnp 2>/dev/null | grep -q ':8010 ' && echo 'WARNING: port 8010 already in use' || echo 'port 8010 free'
    [ -e '$DIR' ] && { echo 'NOTE: $DIR exists:'; ls '$DIR'; } || echo '$DIR is free'"
  ;;

push)
  [ -d "$LOCAL/remote_runs/$RUN/best" ] || { echo "no model at remote_runs/$RUN/best"; exit 1; }
  "${SSH[@]}" "mkdir -p '$DIR'/{models/$RUN,results,data/10k/test,data/10k/val,data/bdd100k_seg_maps/labels}"
  # code
  (cd "$LOCAL" && find ag app scripts deploy -type f \( -name '*.py' -o -name '*.html' -o -name '*.js' -o -name '*.json' \
      -o -name '*.webmanifest' -o -name '*.png' -o -name '*.jpg' -o -name '*.sh' -o -name '*.txt' \) \
      -not -path '*/__pycache__/*' > "$TMP/ag_code")
  (cd "$LOCAL" && printf '%s\n' data/splits.json data/gps_test.json results/seg_baseline_test.json \
      results/audit_baseline_test.json results/seg_baseline_holdout.json results/rule_review.json; \
      if [ -f data/gallery_exclude.txt ]; then echo data/gallery_exclude.txt; fi) >> "$TMP/ag_code"
  "${RSYNC[@]}" --files-from="$TMP/ag_code" "$LOCAL/" "$HOST:$DIR/"
  # model + its training log (the report page draws the curves from it)
  "${RSYNC[@]}" "$LOCAL/remote_runs/$RUN/best" "$LOCAL/remote_runs/$RUN/log.jsonl" "$HOST:$DIR/models/$RUN/"
  # test-set scores measured on the training GPU for this exact model (re-measuring on CPU takes hours)
  "${RSYNC[@]}" "$LOCAL/remote_runs/$RUN/seg_${RUN}_test.json" "$HOST:$DIR/results/seg_finetuned_test.json"
  "${RSYNC[@]}" "$LOCAL/remote_runs/$RUN/audit_${RUN}_test.json" "$HOST:$DIR/results/audit_finetuned_test.json"
  # images the server needs: test frames with US GPS (map) + labelled val set (eval, gallery)
  (cd "$LOCAL" && .venv/bin/python -c "import json; print('\n'.join(f'{k}.jpg' for k in json.load(open('data/gps_test.json'))))") > "$TMP/ag_map"
  "${RSYNC[@]}" --files-from="$TMP/ag_map" "$LOCAL/data/10k/test/" "$HOST:$DIR/data/10k/test/"
  "${RSYNC[@]}" "$LOCAL/data/10k/val/" "$HOST:$DIR/data/10k/val/"
  "${RSYNC[@]}" "$LOCAL/data/bdd100k_seg_maps/labels/val" "$HOST:$DIR/data/bdd100k_seg_maps/labels/"
  "${SSH[@]}" "cd '$DIR' && du -sh . && echo test images: \$(ls data/10k/test | wc -l)"
  ;;

setup)
  "${SSH[@]}" "set -e; cd '$DIR'
    export DEBIAN_FRONTEND=noninteractive
    # swap as a safety net on small-RAM machines (the model process peaks near 3 GB)
    if [ \$(free -m | awk '/Mem:/{print \$2}') -lt 8000 ] && ! swapon --show | grep -q /swapfile; then
      fallocate -l 4G /swapfile && chmod 600 /swapfile && mkswap /swapfile >/dev/null && swapon /swapfile
      grep -q /swapfile /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
    fi
    free -m | sed -n '2,3p'
    export PATH=\$HOME/.local/bin:\$PATH
    command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; }
    uv venv -q --allow-existing --python 3.12 .venv
    if command -v nvidia-smi >/dev/null; then
      uv pip install -q --python .venv/bin/python torch torchvision
    else  # CPU-only wheels are ~10x smaller
      uv pip install -q --python .venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cpu
    fi
    uv pip install -q --python .venv/bin/python -r deploy/requirements-server.txt
    .venv/bin/python -c 'import torch; print(\"torch\", torch.__version__, \"cuda\" if torch.cuda.is_available() else \"cpu\")'
    if [ -n '$DOMAIN' ]; then  # Caddy: automatic Let's Encrypt HTTPS for the domain
      rm -f /etc/apt/sources.list.d/caddy-stable.list  # Caddy's own apt repo now answers 402
      if ! command -v caddy >/dev/null; then
        apt-get update >/dev/null && apt-get install -y caddy >/dev/null || true   # Ubuntu universe package
      fi
      if ! command -v caddy >/dev/null; then  # fallback: official static binary + minimal service
        arch=\$(dpkg --print-architecture)
        curl -fsSL \"https://caddyserver.com/api/download?os=linux&arch=\$arch\" -o /usr/bin/caddy && chmod +x /usr/bin/caddy
        id caddy >/dev/null 2>&1 || useradd --system --home /var/lib/caddy --create-home --shell /usr/sbin/nologin caddy
        mkdir -p /etc/caddy && touch /etc/caddy/Caddyfile
        printf '[Unit]\\nDescription=Caddy\\nAfter=network-online.target\\n[Service]\\nUser=caddy\\nExecStart=/usr/bin/caddy run --config /etc/caddy/Caddyfile\\nExecReload=/usr/bin/caddy reload --config /etc/caddy/Caddyfile\\nAmbientCapabilities=CAP_NET_BIND_SERVICE\\nRestart=always\\n[Install]\\nWantedBy=multi-user.target\\n' > /etc/systemd/system/caddy.service
        systemctl daemon-reload
      fi
      caddy version
      if command -v ufw >/dev/null; then ufw allow 22/tcp >/dev/null; ufw allow 80/tcp >/dev/null; ufw allow 443/tcp >/dev/null; ufw --force enable >/dev/null; ufw status | head -6; fi
    elif ! command -v cloudflared >/dev/null; then
      arch=\$(dpkg --print-architecture)
      curl -fsSL -o /tmp/cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-\$arch.deb
      dpkg -i /tmp/cloudflared.deb
    fi
    echo setup done"
  ;;

rebuild)
  # Map + gallery computed on the server with the deployed model. With a GPU (or AG_FULL=1) the
  # test-set evaluation is re-run too; on CPU the GPU-measured scores from push are kept.
  "${SSH[@]}" "set -e; cd '$DIR'; export HF_HOME=\$PWD/.cache/hf; M=models/$RUN/best; PY=.venv/bin/python
    mkdir -p logs; L=logs/rebuild.log; : > \$L
    if command -v nvidia-smi >/dev/null || [ '${AG_FULL:-0}' = 1 ]; then
      \$PY scripts/eval_seg.py --model \$M --name finetuned --tta --save-preds >> \$L 2>&1
      \$PY scripts/eval_audit.py --name finetuned >> \$L 2>&1
      TTA=--tta
    else
      echo 'CPU host: keeping GPU-measured test scores; map single-pass' | tee -a \$L; TTA=
    fi
    \$PY scripts/make_gallery.py --name finetuned --model \$M >> \$L 2>&1
    \$PY scripts/predict_map.py --model \$M \$TTA >> \$L 2>&1
    rm -rf results/pred_finetuned_test
    grep -E 'mIoU|spearman|support|done:|tp|CPU host' \$L | grep -v -i warn"
  ;;

login)
  # gate answers + shared password are hashed HERE; only the hashes travel to the server
  : "${AG_GATE_QUESTIONS:?set AG_GATE_QUESTIONS (questions separated by |) and AG_GATE_ANSWERS (yes/no list)}"
  : "${AG_GATE_ANSWERS:?set AG_GATE_ANSWERS, e.g. yes,no,yes,yes}"
  lines=$(cd "$LOCAL" && .venv/bin/python scripts/set_password.py --questions "$AG_GATE_QUESTIONS" --answers "$AG_GATE_ANSWERS")
  printf '%s\nAG_SECURE_COOKIE=1\n' "$lines" | "${SSH[@]}" "set -e; mkdir -p /etc/saga; chmod 700 /etc/saga; touch /etc/saga/env
    grep -v -E '^(AG_GATE_QUESTIONS|AG_GATE_HASH|AG_PASSWORD_HASH|AG_SESSION_SECRET|AG_SECURE_COOKIE)=' /etc/saga/env > /etc/saga/env.new || true
    cat >> /etc/saga/env.new; mv /etc/saga/env.new /etc/saga/env; chmod 600 /etc/saga/env
    systemctl is-active --quiet saga-api && systemctl restart saga-api || true
    echo 'login configured (every device must sign in again)'"
  ;;

start)
  "${SSH[@]}" "set -e; cd '$DIR'
    mkdir -p /etc/saga; chmod 700 /etc/saga
    grep -q '^AG_PASSWORD_HASH=' /etc/saga/env 2>/dev/null || { echo 'run: deploy/deploy.sh login  (first)'; exit 1; }
    if ! grep -q '^AG_MODEL=' /etc/saga/env; then
      printf 'AG_MODEL=%s\nAG_ADMIN_TOKEN=%s\nHF_HOME=%s\nAG_TTA=%s\n' '$DIR/models/$RUN/best' \"\$(openssl rand -hex 24)\" '$DIR/.cache/hf' \"\$(command -v nvidia-smi >/dev/null && echo 1 || echo 0)\" >> /etc/saga/env
    fi
    cat > /etc/systemd/system/saga-api.service <<UNIT
[Unit]
Description=SAGA API, platform and mobile app
After=network-online.target
[Service]
WorkingDirectory=$DIR
EnvironmentFile=/etc/saga/env
ExecStart=$DIR/.venv/bin/python -m uvicorn app.server:app --host 127.0.0.1 --port 8010 --proxy-headers --forwarded-allow-ips 127.0.0.1
Restart=always
RestartSec=3
[Install]
WantedBy=multi-user.target
UNIT
    cat > /etc/systemd/system/saga-tunnel.service <<UNIT
[Unit]
Description=SAGA HTTPS quick tunnel (Cloudflare)
After=saga-api.service
[Service]
ExecStart=/usr/bin/cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8010
Restart=always
RestartSec=5
[Install]
WantedBy=multi-user.target
UNIT
    systemctl daemon-reload
    systemctl enable --now saga-api
    if [ -n '$DOMAIN' ]; then
      cat > /etc/caddy/Caddyfile <<CADDY
$DOMAIN {
	encode gzip
	request_body {
		max_size 25MB
	}
	header Strict-Transport-Security \"max-age=31536000\"
	reverse_proxy 127.0.0.1:8010
}
www.$DOMAIN {
	redir https://$DOMAIN{uri} permanent
}
CADDY
      systemctl disable --now saga-tunnel 2>/dev/null || true
      systemctl enable caddy && systemctl reload-or-restart caddy
    else
      systemctl enable --now saga-tunnel
    fi
    for i in \$(seq 1 60); do curl -sf http://127.0.0.1:8010/api/info >/dev/null && break; sleep 2; done
    curl -s http://127.0.0.1:8010/api/info | head -c 160; echo"
  "$0" url
  ;;

url)
  if [ -n "$DOMAIN" ]; then
    echo "platform:   https://$DOMAIN"; echo "mobile app: https://$DOMAIN/m/"
    curl -s -o /dev/null -w "HTTPS check: %{http_code} (303 = sent to the login gate, as expected)\n" "https://$DOMAIN/" || echo "HTTPS not answering yet (DNS or certificate still pending)"
    exit 0
  fi
  "${SSH[@]}" "for i in \$(seq 1 30); do u=\$(journalctl -u saga-tunnel --no-pager -n 200 | grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' | tail -1); [ -n \"\$u\" ] && break; sleep 2; done
    echo \"platform:   \$u\"; echo \"mobile app: \$u/m/\""
  ;;

status)
  "${SSH[@]}" "systemctl --no-pager --lines=0 status saga-api caddy saga-tunnel 2>/dev/null | grep -E '●|Active:' || true
    journalctl -u saga-api --no-pager -n 5"
  ;;

voice)
  # The key is typed hidden here and sent over SSH stdin: never in argv, chat, or shell history.
  read -r -s -p "ElevenLabs API key: " key; echo
  [ -n "$key" ] || { echo "no key entered"; exit 1; }
  { echo "AG_ELEVENLABS_KEY=$key"
    [ -n "${AG_VOICE_ID:-}" ] && echo "AG_ELEVENLABS_VOICE=$AG_VOICE_ID"
    [ -n "${AG_VOICE_MODEL:-}" ] && echo "AG_ELEVENLABS_MODEL=$AG_VOICE_MODEL"; } | "${SSH[@]}" "set -e
    grep -v -E '^AG_ELEVENLABS_(KEY|VOICE|MODEL)=' /etc/saga/env > /etc/saga/env.new || true
    cat >> /etc/saga/env.new; mv /etc/saga/env.new /etc/saga/env; chmod 600 /etc/saga/env
    systemctl restart saga-api
    for i in \$(seq 1 60); do curl -sf -o /dev/null http://127.0.0.1:8010/gate && break; sleep 2; done
    grep -E '^AG_ELEVENLABS_(VOICE|MODEL)=' /etc/saga/env || echo 'voice/model: defaults (George, eleven_multilingual_v2)'
    echo \"voice enabled: \$(grep -c '^AG_ELEVENLABS_KEY=' /etc/saga/env)\""
  ;;

*)
  sed -n '2,20p' "$0"
  ;;
esac
