"""Standalone offline Web-HUD HTTP server for pib.Pro telemetry."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING
from urllib.parse import urlparse

# ruff: noqa: E501

if TYPE_CHECKING:
    from pib_pro.telemetry_gateway.gateway import NonBlockingTelemetryGateway


def render_hud_html() -> str:
    """Self-contained HTML5 diagnostic dashboard (no external network resources)."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>pib.Pro Telemetry HUD</title>
<style>
:root { --bg:#0d1117; --fg:#e6edf3; --accent:#58a6ff; --muted:#8b949e; --ok:#3fb950; }
* { box-sizing:border-box; }
body { margin:0; font-family:system-ui,sans-serif; background:var(--bg); color:var(--fg); }
header { padding:12px 16px; border-bottom:1px solid #30363d; display:flex; gap:12px; align-items:center; }
h1 { font-size:1.1rem; margin:0; }
.badge { padding:4px 10px; border-radius:999px; background:#21262d; font-size:0.85rem; }
.grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; padding:12px; }
.panel { background:#161b22; border:1px solid #30363d; border-radius:8px; padding:10px; }
.panel h2 { margin:0 0 8px; font-size:0.95rem; color:var(--accent); }
table { width:100%; border-collapse:collapse; font-size:0.8rem; }
th,td { border:1px solid #30363d; padding:4px 6px; text-align:left; }
canvas,svg { width:100%; background:#0d1117; border-radius:4px; }
.meter { height:8px; background:#21262d; border-radius:4px; overflow:hidden; }
.meter > span { display:block; height:100%; background:var(--accent); width:0%; }
.compass { width:120px; height:120px; margin:auto; }
footer { padding:8px 16px; color:var(--muted); font-size:0.75rem; }
</style>
</head>
<body>
<header>
<h1>pib.Pro REP-103 HUD</h1>
<span class="badge" id="ja-badge">Joint Attention: —</span>
<span class="badge" id="bt-badge">BT: —</span>
</header>
<div class="grid">
<section class="panel">
<h2>Scene objects (REP-103)</h2>
<table><thead><tr><th>ID</th><th>Class</th><th>X</th><th>Y</th><th>Z</th><th>Conf</th></tr></thead>
<tbody id="scene-rows"></tbody></table>
</section>
<section class="panel">
<h2>Workbench (top-down)</h2>
<canvas id="bench-canvas" width="420" height="280" aria-label="Workbench top-down"></canvas>
</section>
<section class="panel">
<h2>Gaze &amp; pointing</h2>
<p id="gaze-line">Gaze ray: —</p>
<p id="point-line">Pointing ray: —</p>
</section>
<section class="panel">
<h2>H-JEPA Surprise</h2>
<div class="meter"><span id="surprise-bar"></span></div>
<p id="surprise-val">Surprise_t: 0</p>
</section>
<section class="panel">
<h2>Seeed ReSpeaker 360° DOA</h2>
<svg class="compass" viewBox="0 0 100 100" id="doa-compass">
<circle cx="50" cy="50" r="45" fill="none" stroke="#30363d"/>
<line x1="50" y1="50" x2="50" y2="10" stroke="#58a6ff" id="doa-needle"/>
<text x="50" y="8" text-anchor="middle" fill="#8b949e" font-size="6">N</text>
</svg>
<p id="vad-ind">VAD: inactive</p>
</section>
<section class="panel">
<h2>Chest OAK-4 D Pro</h2>
<p id="oak-status">Status: —</p>
</section>
<section class="panel">
<h2>Jetson Thor thermal / power</h2>
<p id="thor-health">—</p>
</section>
</div>
<footer>Offline HUD — polls /api/v1/telemetry</footer>
<script>
(function(){
var surpriseHistory=[];
function drawBench(markers){
  var c=document.getElementById("bench-canvas");
  if(!c||!c.getContext) return;
  var ctx=c.getContext("2d");
  var w=c.width,h=c.height;
  ctx.fillStyle="#0d1117"; ctx.fillRect(0,0,w,h);
  ctx.strokeStyle="#30363d";
  for(var gx=0;gx<=4;gx++){
    var x=gx*w/4; ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,h); ctx.stroke();
  }
  for(var gy=0;gy<=4;gy++){
    var y=gy*h/4; ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(w,y); ctx.stroke();
  }
  ctx.fillStyle="#58a6ff";
  (markers||[]).forEach(function(m){
    var px=((m.position_xyz[0]-0.15)/0.5)*w;
    var py=h-((m.position_xyz[1]+0.35)/0.7)*h;
    ctx.beginPath(); ctx.arc(px,py,4,0,Math.PI*2); ctx.fill();
  });
}
function setDoa(azRad,vad){
  var needle=document.getElementById("doa-needle");
  if(needle){
    var deg=azRad*180/Math.PI;
    var rad=(deg-90)*Math.PI/180;
    var x2=50+40*Math.cos(rad), y2=50+40*Math.sin(rad);
    needle.setAttribute("x2",x2); needle.setAttribute("y2",y2);
  }
  var v=document.getElementById("vad-ind");
  if(v) v.textContent="VAD: "+(vad?"active":"inactive");
}
function applyPayload(d){
  var sum=d.world_state_summary||{};
  document.getElementById("ja-badge").textContent=
    "Joint Attention: "+(sum.joint_attention_object_id>=0?("#"+sum.joint_attention_object_id):"none");
  document.getElementById("bt-badge").textContent="BT node: "+(d.active_bt_node_id||0);
  var rows=document.getElementById("scene-rows");
  rows.innerHTML="";
  (d.scene_markers_rep103||[]).forEach(function(m){
    var tr=document.createElement("tr");
    tr.innerHTML="<td>"+m.object_id+"</td><td>"+m.class_id+"</td><td>"+
      m.position_xyz[0].toFixed(3)+"</td><td>"+m.position_xyz[1].toFixed(3)+"</td><td>"+
      m.position_xyz[2].toFixed(3)+"</td><td>"+m.confidence.toFixed(2)+"</td>";
    rows.appendChild(tr);
  });
  drawBench(d.scene_markers_rep103);
  var s=d.surprise_scalar||0;
  document.getElementById("surprise-val").textContent="Surprise_t: "+s.toFixed(3);
  document.getElementById("surprise-bar").style.width=Math.min(100,s*100)+"%";
  var mic=d.seeed_mic_status||{};
  setDoa(mic.doa_azimuth_rad||0,!!mic.vad_active);
  var cam=d.chest_oak4_status||{};
  document.getElementById("oak-status").textContent=
    "Frame "+(cam.frame||"?")+" streaming="+cam.streaming+" depth_ok="+cam.depth_ok;
  var h=d.system_health||{};
  document.getElementById("thor-health").textContent=
    "CPU "+h.cpu_util_pct+"% GPU "+h.gpu_util_pct+"% Tmax "+
    h.thermal_max_c+"°C power headroom "+h.power_headroom_pct+"%";
  document.getElementById("gaze-line").textContent="Gaze ray (bus): human_present="+!!sum.human_present;
  document.getElementById("point-line").textContent="Objects tracked: "+(sum.object_count||0);
}
function poll(){
  fetch("/api/v1/telemetry").then(function(r){return r.json();}).then(applyPayload).catch(function(){});
}
poll();
setInterval(poll,1000);
})();
</script>
</body>
</html>"""


class TelemetryHudServer:
    """Threading HTTP server exposing offline HUD and JSON telemetry API."""

    def __init__(
        self,
        gateway: NonBlockingTelemetryGateway,
        host: str = "127.0.0.1",
        port: int = 8765,
    ) -> None:
        self._gateway = gateway
        self._host = host
        self._port = port
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        gateway_ref = gateway

        class _HudHandler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:
                _ = (format, args)

            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                path = parsed.path.rstrip("/") or "/"
                if path == "/":
                    body = render_hud_html().encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if path == "/api/v1/health":
                    body = json.dumps({"ok": True, "service": "pib-pro-hud"}).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if path == "/api/v1/telemetry":
                    snap = gateway_ref.collect(time.monotonic())
                    payload_json = gateway_ref.publish_json(snap)
                    body = payload_json.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self.send_response(404)
                self.end_headers()

        self._handler_cls = _HudHandler

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port

    @property
    def base_url(self) -> str:
        return f"http://{self._host}:{self._port}/"

    def start_background(self) -> None:
        if self._httpd is not None:
            return
        self._httpd = ThreadingHTTPServer((self._host, self._port), self._handler_cls)
        self._port = int(self._httpd.server_address[1])
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._httpd is None:
            return
        self._httpd.shutdown()
        self._httpd.server_close()
        self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
