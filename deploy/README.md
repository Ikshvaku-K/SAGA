# Deploying SAGA (Street Accessibility and Guidance Audit)

One server runs everything: the platform (`/`), the iPhone app (`/m/`), and the report API. All model inference happens on the server; phones only upload photos.

```
iPhone app (/m/) ─┐
Platform (/)     ─┼─ HTTPS ─► Caddy ──► saga-api   uvicorn on 127.0.0.1:8010
                                         SegFormer-B2 (GPU-trained run b2_8k), audit, reports
```

## Steps

**DNS first** (at your domain's DNS provider), both pointing to the Vultr server's IP:

| Type | Name | Value |
|---|---|---|
| A | `@` (sagatech.us) | server IP |
| A | `www` | server IP |

```bash
export AG_HOST=root@<server-ip>              # AG_DOMAIN defaults to sagatech.us
deploy/deploy.sh check      # OS, GPU, disk, port 8010, whether /opt/saga exists. Read-only
deploy/deploy.sh push       # code + model + metrics + images for the map/gallery (~350 MB)
deploy/deploy.sh setup      # venv (CUDA PyTorch if a GPU is present, else CPU), Caddy, firewall, Node 22
deploy/deploy.sh rebuild    # test-set eval, audit scores, gallery, city map, computed on the server
AG_GATE_QUESTIONS='Q1?|Q2?|Q3?|Q4?' AG_GATE_ANSWERS=yes,no,yes,yes deploy/deploy.sh login   # password asked on YOUR machine; only hashes are sent
deploy/deploy.sh start      # systemd services + Caddy HTTPS for sagatech.us
```

Later: `deploy/deploy.sh status`, `deploy/deploy.sh url`, `deploy/deploy.sh login` again to change the password (this logs every device out).

The app listens on `127.0.0.1` only; Caddy serves the public HTTPS on ports 80/443 and gets the certificate automatically. Secrets live in `/etc/saga/env` (mode 600): the gate and password **hashes**, the session signing key, and the admin (moderation) token.

## Login

Every page, photo and API call needs a session. Visitors go to `/gate`, choose **User** (the yes/no questions from `AG_GATE_QUESTIONS`, then the shared password) or **Demo** (three fixed questions and a first name, 15 minutes). The correct answers and the password are checked only on the server, against salted hashes.

- A wrong answer or password shows "Please try again" and starts over, without revealing which part was wrong.
- **5 failed attempts per IP address** lock that IP out for 15 minutes.
- A successful login sets a signed, HttpOnly cookie valid for **30 days**.
- On iPhone, the home-screen app keeps its own storage separate from Safari, so you sign in once inside the app itself.

## HTTPS without a domain (fallback: `AG_DOMAIN= deploy/deploy.sh start`)

| Option | URL | Stable? | Needs |
|---|---|---|---|
| **Cloudflare quick tunnel** (default) | `https://<random>.trycloudflare.com` | **No.** It changes whenever the tunnel service restarts. | Nothing |
| **Tailscale Funnel** | `https://<server>.<tailnet>.ts.net` | Yes | Free Tailscale account |
| Cloudflare named tunnel | your own hostname | Yes | A domain on Cloudflare |

The iPhone app is pinned to its URL once added to the home screen, so for everyday use switch to Tailscale Funnel. On the server:

```bash
curl -fsSL https://tailscale.com/install.sh | sh
tailscale up            # opens a login link; sign in with your Tailscale account
tailscale funnel --bg 8010
systemctl disable --now saga-tunnel
```

## Installing the iPhone app

1. Open `https://sagatech.us/m/` in **Safari** on the iPhone and sign in.
2. Tap **Share**, then **Add to Home Screen**, then **Add**.
3. Open SAGA from the home screen, tap **Get started**, then **Take photo**. iOS asks once for camera and location access.

The app downsizes each photo on the phone, uploads it, and shows the findings as text, then the overlay, then 👍 / 👎.

## Abuse limits

- **Uploads:** at most 30 per IP address per 10 minutes (`AG_APP_LIMIT`), 20 MB maximum.
- **Editing reports:** each report has its own random edit key, so only the phone that sent it can add its location or feedback.
- **Removing reports:** `DELETE /api/reports/{id}` with the `X-Admin-Token` (the `AG_ADMIN_TOKEN` value in `/etc/saga/env`).
