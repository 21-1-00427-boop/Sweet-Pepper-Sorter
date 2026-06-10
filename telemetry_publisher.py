"""
telemetry_publisher.py
──────────────────────
Publishes the pepper-sorter TALLY to an MQTT broker whenever a count changes.

Pipeline:
    this module ──MQTT──► Mosquitto ──► Node-RED ──► InfluxDB ──► Grafana

Design notes
────────────
• Publish-on-change only: call publish_tally(stats) after any count update;
  it diffs against the last payload and only sends if something changed.
• Degrades gracefully: if the broker is unreachable the sorter keeps running;
  publishing is best-effort and never raises into the control loop.
• One retained topic so a freshly-started Node-RED/Grafana immediately sees
  the latest totals without waiting for the next change.

Requires: paho-mqtt  →  pip install paho-mqtt --break-system-packages
"""

import json
import time
import threading

try:
    import paho.mqtt.client as mqtt
except Exception:                       # pragma: no cover
    mqtt = None


# ── Configuration ───────────────────────────────────────────────────────────
MQTT_HOST   = "localhost"   # broker runs on the Pi
MQTT_PORT   = 1883
MQTT_TOPIC  = "pepper/tally"
MQTT_QOS    = 0
RETAIN      = True          # keep last tally on the broker for late subscribers
CLIENT_ID   = "pepper_sorter"

# Which stat keys form the "tally" we care about. Only changes to these
# trigger a publish.
TALLY_KEYS = [
    "total_accepted",
    "total_rejected",
]
# Per-bin counts (nested dict) are always included and diffed too.


class TelemetryPublisher:
    """Thin, fault-tolerant MQTT publisher for the sorter tally."""

    def __init__(self, host=MQTT_HOST, port=MQTT_PORT, topic=MQTT_TOPIC):
        self._host  = host
        self._port  = port
        self._topic = topic
        self._lock  = threading.Lock()
        self._last_payload = None        # last tally dict sent (for diffing)
        self._client = None
        self._connected = False

        if mqtt is None:
            print("[MQTT] paho-mqtt not installed — telemetry disabled. "
                  "Install with: pip install paho-mqtt --break-system-packages")
            return

        try:
            # paho-mqtt 2.x requires the callback API version argument
            try:
                self._client = mqtt.Client(
                    client_id=CLIENT_ID,
                    callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                )
            except (AttributeError, TypeError):
                # paho-mqtt 1.x fallback
                self._client = mqtt.Client(client_id=CLIENT_ID)

            self._client.on_connect    = self._on_connect
            self._client.on_disconnect = self._on_disconnect
            # Connect asynchronously so a missing broker never blocks startup
            self._client.connect_async(self._host, self._port, keepalive=60)
            self._client.loop_start()
            print(f"[MQTT] Connecting to {self._host}:{self._port} "
                  f"(topic '{self._topic}')...")
        except Exception as e:
            print(f"[MQTT] Init failed ({e}); telemetry disabled.")
            self._client = None

    # ── Callbacks ────────────────────────────────────────────────────────────

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        self._connected = True
        print(f"[MQTT] Connected to broker (topic '{self._topic}').")

    def _on_disconnect(self, client, userdata, *args):
        self._connected = False
        print("[MQTT] Disconnected from broker (will auto-reconnect).")

    # ── Publishing ────────────────────────────────────────────────────────────

    def _build_tally(self, stats: dict) -> dict:
        """Extract just the tally fields from the full sorter_stats dict."""
        tally = {k: stats.get(k, 0) for k in TALLY_KEYS}
        # Per-bin deposited counts
        tally["bins"] = dict(stats.get("bins", {}))
        return tally

    def publish_tally(self, stats: dict):
        """
        Publish the current tally if (and only if) it changed since last time.
        Safe to call frequently; cheap when nothing changed.
        """
        if self._client is None:
            return
        tally = self._build_tally(stats)

        with self._lock:
            if tally == self._last_payload:
                return                       # no change → don't publish
            self._last_payload = tally

        payload = json.dumps({
            "ts":             int(time.time() * 1000),  # epoch ms
            "total_accepted": tally["total_accepted"],
            "total_rejected": tally["total_rejected"],
            "bins":           tally["bins"],
        })

        try:
            self._client.publish(self._topic, payload, qos=MQTT_QOS, retain=RETAIN)
            print(f"[MQTT] Tally published → {payload}")
        except Exception as e:
            print(f"[MQTT] Publish failed ({e}); will retry on next change.")

    def close(self):
        if self._client is not None:
            try:
                self._client.loop_stop()
                self._client.disconnect()
            except Exception:
                pass
