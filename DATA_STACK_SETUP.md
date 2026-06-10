# Pepper Sorter — Data Stack Setup (MQTT → Node-RED → InfluxDB → Grafana)

All components run on the Raspberry Pi. The Python sorter publishes the tally
to MQTT only when a count changes; Node-RED writes it to InfluxDB; Grafana
charts it.

```
camera_system.py ──MQTT──► Mosquitto ──► Node-RED ──► InfluxDB 2.x ──► Grafana
   topic: pepper/tally
```

────────────────────────────────────────────────────────────────────────
## 0. Prerequisites

InfluxDB 2.x needs a **64-bit OS**. Check:
```bash
uname -m        # aarch64 / arm64 = 64-bit (good for InfluxDB 2.x)
                # armv7l = 32-bit  → use InfluxDB 1.8 instead (see note at end)
```

Install the Python MQTT client:
```bash
pip install paho-mqtt --break-system-packages
```

────────────────────────────────────────────────────────────────────────
## 1. Mosquitto (MQTT broker)

```bash
sudo apt update
sudo apt install -y mosquitto mosquitto-clients
sudo systemctl enable mosquitto
sudo systemctl start mosquitto
```

Test it (in two terminals):
```bash
mosquitto_sub -t 'pepper/tally'                 # terminal 1: listen
mosquitto_pub -t 'pepper/tally' -m 'hello'      # terminal 2: send
```
You should see "hello" appear in terminal 1.

The sorter publishes to `localhost:1883`, topic `pepper/tally`, with retain on.

────────────────────────────────────────────────────────────────────────
## 2. InfluxDB 2.x

```bash
# Add the InfluxData repo
curl -s https://repos.influxdata.com/influxdata-archive_compat.key | gpg --dearmor | sudo tee /etc/apt/trusted.gpg.d/influxdata.gpg >/dev/null
echo 'deb https://repos.influxdata.com/debian stable main' | sudo tee /etc/apt/sources.list.d/influxdata.list
sudo apt update
sudo apt install -y influxdb2
sudo systemctl enable influxdb
sudo systemctl start influxdb
```

Open `http://<pi-ip>:8086` and complete the setup wizard:
- **Org:** `pepper`
- **Bucket:** `pepper_sorter`
- Save the **API token** it generates — Node-RED and Grafana both need it.

────────────────────────────────────────────────────────────────────────
## 3. Node-RED

```bash
bash <(curl -sL https://raw.githubusercontent.com/node-red/linux-installers/master/deb/update-nodejs-and-nodered)
sudo systemctl enable nodered
sudo systemctl start nodered
```

Open `http://<pi-ip>:1880`. Install the InfluxDB nodes:
- Menu → Manage palette → Install → search `node-red-contrib-influxdb` → install.

Then **import the flow** below (Menu → Import → paste → Import):

```json
[
  {
    "id": "mqtt_in_tally",
    "type": "mqtt in",
    "name": "pepper/tally",
    "topic": "pepper/tally",
    "qos": "0",
    "broker": "mqtt_broker_local",
    "x": 140, "y": 120,
    "wires": [["json_parse"]]
  },
  {
    "id": "json_parse",
    "type": "json",
    "name": "parse JSON",
    "x": 330, "y": 120,
    "wires": [["build_point"]]
  },
  {
    "id": "build_point",
    "type": "function",
    "name": "build Influx point",
    "func": "var d = msg.payload;\nvar bins = d.bins || {};\nvar fields = {\n  total_accepted: d.total_accepted || 0,\n  total_rejected: d.total_rejected || 0\n};\n// flatten each bin into its own field (spaces -> underscores)\nfor (var k in bins) {\n  fields['bin_' + k.replace(/ /g,'_')] = bins[k];\n}\nmsg.payload = [fields, { source: 'pepper_sorter' }];\nreturn msg;",
    "outputs": 1,
    "x": 540, "y": 120,
    "wires": [["influx_out"]]
  },
  {
    "id": "influx_out",
    "type": "influxdb out",
    "name": "pepper_sorter bucket",
    "influxdb": "influx_local",
    "org": "pepper",
    "bucket": "pepper_sorter",
    "measurement": "tally",
    "precision": "ms",
    "x": 770, "y": 120,
    "wires": []
  },
  {
    "id": "mqtt_broker_local",
    "type": "mqtt-broker",
    "name": "localhost",
    "broker": "localhost",
    "port": "1883",
    "keepalive": "60"
  },
  {
    "id": "influx_local",
    "type": "influxdb",
    "name": "InfluxDB 2 local",
    "influxdbVersion": "2.0",
    "url": "http://localhost:8086"
  }
]
```

After import:
1. Double-click the **InfluxDB 2 local** config node → paste your API **token**,
   confirm org `pepper`, URL `http://localhost:8086`.
2. Click **Deploy**.

The function node flattens the JSON: `total_accepted`, `total_rejected`, and one
field per bin (e.g. `bin_Red_Large`, `bin_Mix_Medium`, `bin_Reject`).

────────────────────────────────────────────────────────────────────────
## 4. Grafana

```bash
sudo apt install -y apt-transport-https software-properties-common
curl https://apt.grafana.com/gpg.key | gpg --dearmor | sudo tee /etc/apt/trusted.gpg.d/grafana.gpg >/dev/null
echo "deb https://apt.grafana.com stable main" | sudo tee /etc/apt/sources.list.d/grafana.list
sudo apt update
sudo apt install -y grafana
sudo systemctl enable grafana-server
sudo systemctl start grafana-server
```

Open `http://<pi-ip>:3000` (default login admin/admin).

Add the data source: **Connections → Data sources → Add → InfluxDB**
- Query language: **Flux**
- URL: `http://localhost:8086`
- Organization: `pepper`
- Token: your API token
- Default bucket: `pepper_sorter`

### Example panels (Flux queries)

**Accepted vs Rejected over time (time series):**
```flux
from(bucket: "pepper_sorter")
  |> range(start: -6h)
  |> filter(fn: (r) => r._measurement == "tally")
  |> filter(fn: (r) => r._field == "total_accepted" or r._field == "total_rejected")
```

**Current per-bin totals (bar gauge — use "Last" value):**
```flux
from(bucket: "pepper_sorter")
  |> range(start: -24h)
  |> filter(fn: (r) => r._measurement == "tally")
  |> filter(fn: (r) => r._field =~ /^bin_/)
  |> last()
```

**Total sorted (stat panel):**
```flux
from(bucket: "pepper_sorter")
  |> range(start: -24h)
  |> filter(fn: (r) => r._measurement == "tally" and r._field == "total_accepted")
  |> last()
```

────────────────────────────────────────────────────────────────────────
## Data schema reference

- **Measurement:** `tally`
- **Tag:** `source = pepper_sorter`
- **Fields:**
  - `total_accepted` (int)
  - `total_rejected` (int)
  - `bin_Reject`, `bin_Green_Large`, `bin_Green_Medium`, `bin_Red_Large`,
    `bin_Red_Medium`, `bin_Orange_Large`, `bin_Orange_Medium`,
    `bin_Mix_Large`, `bin_Mix_Medium` (int each)
- **Timestamp:** publish time (ms precision), carried as `ts` in the payload.

The MQTT payload looks like:
```json
{
  "ts": 1733300000000,
  "total_accepted": 12,
  "total_rejected": 3,
  "bins": { "Red Large": 4, "Green Medium": 2, "Reject": 3, ... }
}
```

────────────────────────────────────────────────────────────────────────
## If you are on 32-bit Raspberry Pi OS (InfluxDB 1.8)

```bash
sudo apt install -y influxdb
sudo systemctl enable influxdb && sudo systemctl start influxdb
influx -execute 'CREATE DATABASE pepper_sorter'
```
In Node-RED, set the influxdb config node version to **1.x** and give it the
database name `pepper_sorter` (no token/org). In Grafana, add an InfluxDB data
source with query language **InfluxQL** and database `pepper_sorter`.

────────────────────────────────────────────────────────────────────────
## Startup order

1. Mosquitto, InfluxDB, Node-RED, Grafana (all enabled as services, auto-start)
2. `python main.py`

The publisher connects asynchronously and retries, so the order isn't critical —
if the broker starts after the sorter, the next tally change will connect and
publish. The retained message means Grafana shows the latest totals even after a
restart.
