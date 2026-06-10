"""
stepper_controller.py
─────────────────────
Handles all motor hardware (Motor 1 driver + Motor 2 ULN2003), the limit
switch, AND the two PCA9685 servos that move in sync with the motor phases.

The class exposes a clean interface so main.py can call high-level methods
without worrying about GPIO / servo details.

Servo behaviour (mirrors combined_Stepper_Servo_Manual.py):
  • Phase 1 (M1 moving)   → servos stay in the HOLD (down) posture
  • Phase 2 (M2 running)  → Ch 11 returns to standby immediately; Ch 10 runs an
                            independent sweep parallel to M2
  • Phase 3 (M2 paused)   → both servos snap back DOWN into the hold posture for
                            the inspection pause
"""

from gpiozero import OutputDevice, Button
import time
import math
import threading
import queue

# Servo driver (PCA9685). Optional — runs in simulation if unavailable.
try:
    from adafruit_servokit import ServoKit
except Exception:                     # pragma: no cover
    ServoKit = None

# ── Pin / Motion Configuration ─────────────────────────────────────────────
PUL_PIN            = 17
DIR_PIN            = 27
ENA_PIN            = 22

MOTOR2_PINS        = (23, 24, 25, 8)
RPM                = 10
STEPS_PER_REV      = 4096
MOTOR2_DIRECTION   = "CW"

LIMIT_SWITCH_PIN   = 16
SWITCH_LOCKOUT_TIME = 2.0   # seconds: ignore switch after M2 starts
SWITCH_SUSTAIN_TIME = 0.5   # switch must stay closed this long

PULSES_PER_CYCLE   = 960    # Motor 1: 36° per cycle
CYCLE_PAUSE_TIME   = 5      # seconds pause after M2 stops

# ── Servo Configuration (PCA9685) ───────────────────────────────────────────
SERVO_CH_10       = 10
SERVO_CH_11       = 11
STANDBY_10        = 180.0
STANDBY_11        = 120.0
SERVO_RAMP_STEPS  = 30      # smoothness steps for interpolation
POST_PAUSE_DELAY  = 0.1     # delay for Ch 10 while motor runs
SWEEP_HOLD_S      = 1.0     # hold duration for Ch 10 at its lowest position
HOLD_OFFSET       = 120.0   # degrees below standby for the "hold" posture
CH10_SWEEP_DROP   = 28.0    # degrees Ch 10 sweeps during the M2 worker

# Reject-paddle servo (fires when a CAM 1-rejected pepper reaches the station)
SERVO_CH_6        = 6
STANDBY_6         = 115.0   # standby / holding angle
CH6_SWEEP_DROP    = 60.0    # degrees Ch 6 sweeps down to eject
CH6_SWEEP_DUR     = 0.25    # ramp duration for the down sweep
CH6_RETURN_DUR    = 0.35    # ramp duration back to standby

# ── Sorted-bin gate servos ──────────────────────────────────────────────────
# Each of the 9 bins has its own gate servo. When a pepper reaches that bin's
# cycle (during the pause), the gate sweeps down -BIN_SWEEP_DROP then returns
# to its standby angle, using the same smooth ramp as the reject paddle.
# Keys MUST match the bin labels used by CameraSystem.
BIN_SERVO_CHANNEL = {
    "Reject":        8,
    "Green Large":   3,
    "Green Medium":  4,
    "Red Large":     7,
    "Red Medium":    1,
    "Orange Large":  5,
    "Orange Medium": 9,
    "Mix Large":     0,
    "Mix Medium":    2,
}
BIN_SERVO_STANDBY = {
    "Reject":        106.0,
    "Green Large":   113.0,
    "Green Medium":  105.0,
    "Red Large":     101.0,
    "Red Medium":    106.0,
    "Orange Large":  118.0,
    "Orange Medium":  97.0,
    "Mix Large":     117.0,
    "Mix Medium":     99.0,
}
BIN_SWEEP_DROP    = 60.0    # degrees each bin gate sweeps down
BIN_SWEEP_DUR     = 0.25    # ramp down duration
BIN_RETURN_DUR    = 0.35    # ramp back to standby duration

# Half-step sequence for Motor 2
HALF_STEPS = [
    (1, 0, 0, 0),
    (1, 1, 0, 0),
    (0, 1, 0, 0),
    (0, 1, 1, 0),
    (0, 0, 1, 0),
    (0, 0, 1, 1),
    (0, 0, 0, 1),
    (1, 0, 0, 1),
]


class SortDecision:
    """
    Passed through result_queue by CameraSystem → consumed by StepperController.

    Attributes
    ----------
    pepper_id   : str   – unique ID for traceability (e.g. "CAM1_TRK_0")
    reject      : bool  – True  → reject on next cycle (M2 ejects the pepper)
                          False → accept (M2 stops normally, pepper advances)
    color_stage : str   – dominant colour label from cam 1 (carried to cam 0)
    maturity    : str   – ripeness label from cam 1
    """
    def __init__(self, pepper_id: str, reject: bool,
                 color_stage: str = "Unknown", maturity: str = "Unknown"):
        self.pepper_id   = pepper_id
        self.reject      = reject
        self.color_stage = color_stage
        self.maturity    = maturity

    def __repr__(self):
        action = "REJECT" if self.reject else "ACCEPT"
        return (f"<SortDecision id={self.pepper_id} action={action} "
                f"color={self.color_stage} maturity={self.maturity}>")


class StepperController:
    """
    Manages the two-motor conveyor + limit switch + two synchronised servos.

    Parameters
    ----------
    result_queue : queue.Queue[SortDecision]
        Decisions produced by CameraSystem; consumed here at cycle start.
    status_callback : callable(str) | None
        Optional function called with a human-readable status string so
        the GUI / Flask layer can display live progress.
    """

    def __init__(self, result_queue: queue.Queue, status_callback=None):
        self._result_queue   = result_queue
        self._status_cb      = status_callback or (lambda msg: None)

        # Optional hook: a callable returning True if the Ch 6 reject paddle
        # should fire this pause. Set by main.py via set_paddle_check().
        self._paddle_check   = None

        # Optional hook: a callable returning a list of bin labels whose gate
        # servos should fire this pause. Set by main.py via set_bin_gate_check().
        self._bin_gate_check = None

        # ── Motor / switch hardware ────────────────────────────────────────
        self._pul         = OutputDevice(PUL_PIN)
        self._dir_pin     = OutputDevice(DIR_PIN)
        self._ena         = OutputDevice(ENA_PIN)
        self._m2_pins     = [OutputDevice(p) for p in MOTOR2_PINS]
        self._limit_sw    = Button(LIMIT_SWITCH_PIN, pull_up=True)

        seq = HALF_STEPS if MOTOR2_DIRECTION == "CW" else list(reversed(HALF_STEPS))
        self._sequence    = seq
        self._step_delay  = 60.0 / (RPM * STEPS_PER_REV)

        # ── Servo hardware (PCA9685) ───────────────────────────────────────
        self._kit = None
        if ServoKit is not None:
            try:
                self._kit = ServoKit(channels=16)
                self._kit.servo[SERVO_CH_10].angle = STANDBY_10
                self._kit.servo[SERVO_CH_11].angle = STANDBY_11
                self._kit.servo[SERVO_CH_6].angle  = STANDBY_6
                # All 9 bin gate servos to their standby angles
                for _bin, _ch in BIN_SERVO_CHANNEL.items():
                    self._kit.servo[_ch].angle = BIN_SERVO_STANDBY[_bin]
            except Exception as e:
                print(f"[Warning] Could not initialise PCA9685: {e}. "
                      f"Running servos in simulation mode.")
                self._kit = None
        else:
            print("[Warning] adafruit_servokit not available. "
                  "Running servos in simulation mode.")

        # ── State ──────────────────────────────────────────────────────────
        self._is_running   = False
        self._cycle_count  = 0
        self._pulse_count  = 0
        self._thread       = None

        # Last decision consumed from the queue (used for logging / HTML)
        self.last_decision: SortDecision | None = None

        # Motor phase for the dashboard: "IDLE" | "MOVING" | "PAUSING"
        self.phase = "IDLE"

        # AI gate: set during PAUSING (AI on), cleared during MOVING (AI off)
        self.inference_allowed = threading.Event()

    # ── Public API ─────────────────────────────────────────────────────────

    def set_paddle_check(self, fn):
        """Register a callable () -> bool that says whether to fire Ch 6 now."""
        self._paddle_check = fn

    def set_bin_gate_check(self, fn):
        """Register a callable () -> list[str] of bin labels to fire this pause."""
        self._bin_gate_check = fn

    def start(self, direction: str = "CW"):
        """Begin the auto-cycle loop in a background thread."""
        if self._is_running:
            return
        self._ena.off()                         # Enable M1 driver (LOW = active on DM542)
        self._dir_pin.off() if direction == "CW" else self._dir_pin.on()
        time.sleep(0.05)

        self._is_running = True
        self._thread = threading.Thread(target=self._auto_cycle, daemon=True)
        self._thread.start()

    def stop(self):
        """Request the auto-cycle loop to stop gracefully."""
        self._is_running = False
        self.phase = "IDLE"
        self.inference_allowed.clear()
        self._ena.on()                          # Disable M1 driver
        self._coils_off()
        self._servos_to_standby()
        self._status_cb("Status: Stopped")

    def reset_counters(self):
        """Reset cycle and pulse counters to zero."""
        self._cycle_count = 0
        self._pulse_count = 0

    def get_stats(self) -> dict:
        """Return a snapshot of current counters for display."""
        return {
            "cycle_count":   self._cycle_count,
            "pulse_count":   self._pulse_count,
            "is_running":    self._is_running,
            "phase":         self.phase,
            "last_decision": repr(self.last_decision),
        }

    def close(self):
        """Release all GPIO resources – call on shutdown."""
        self.stop()
        self._servos_to_standby()
        self._pul.close()
        self._dir_pin.close()
        self._ena.close()
        self._limit_sw.close()
        for p in self._m2_pins:
            p.close()

    # ── Servo helpers ────────────────────────────────────────────────────────

    def _servos_to_standby(self):
        if self._kit:
            self._kit.servo[SERVO_CH_10].angle = STANDBY_10
            self._kit.servo[SERVO_CH_11].angle = STANDBY_11
            self._kit.servo[SERVO_CH_6].angle  = STANDBY_6
            for _bin, _ch in BIN_SERVO_CHANNEL.items():
                self._kit.servo[_ch].angle = BIN_SERVO_STANDBY[_bin]

    def fire_bin_gate(self, bin_label):
        """
        Sweep the gate servo for `bin_label` DOWN -BIN_SWEEP_DROP then back to
        its standby angle (smooth ramp). Called when a pepper reaches that bin's
        cycle during the pause.
        """
        ch      = BIN_SERVO_CHANNEL.get(bin_label)
        standby = BIN_SERVO_STANDBY.get(bin_label)
        if ch is None or standby is None:
            print(f"[Servo] Unknown bin '{bin_label}', no gate fired.")
            return
        down = standby - BIN_SWEEP_DROP
        if self._kit is None:
            print(f"[Servo] (sim) Bin gate '{bin_label}' Ch{ch}: "
                  f"{standby}° → {down}° → {standby}°")
            return
        print(f"[Servo] Bin gate '{bin_label}' Ch{ch}: "
              f"{standby}° → {down}° → {standby}°")
        self._move_servo_ramp(ch, standby, down,    duration=BIN_SWEEP_DUR)
        self._move_servo_ramp(ch, down,    standby, duration=BIN_RETURN_DUR)

    def _reject_paddle_sweep(self):
        """
        Ch 6 reject paddle: smooth ramp DOWN -CH6_SWEEP_DROP from standby,
        then immediately ramp back to standby. Fires during the pause of the
        cycle that ejects a CAM 1-rejected pepper.
        """
        if self._kit is None:
            print("[Servo] (sim) Ch6 reject sweep")
            return
        down = STANDBY_6 - CH6_SWEEP_DROP
        print(f"[Servo] Ch6 reject paddle: {STANDBY_6}° → {down}° → {STANDBY_6}°")
        self._move_servo_ramp(SERVO_CH_6, STANDBY_6, down, duration=CH6_SWEEP_DUR)
        self._move_servo_ramp(SERVO_CH_6, down, STANDBY_6, duration=CH6_RETURN_DUR)

    def _move_servos_smooth(self, target_10, target_11, duration=0.5):
        """Linear interpolation of both servos to their targets."""
        if self._kit is None:
            return
        cur_10 = self._kit.servo[SERVO_CH_10].angle
        cur_11 = self._kit.servo[SERVO_CH_11].angle
        if cur_10 is None: cur_10 = STANDBY_10
        if cur_11 is None: cur_11 = STANDBY_11

        steps = SERVO_RAMP_STEPS
        delay = duration / steps
        for i in range(1, steps + 1):
            t = i / steps
            p10 = cur_10 + (target_10 - cur_10) * t
            p11 = cur_11 + (target_11 - cur_11) * t
            self._kit.servo[SERVO_CH_10].angle = max(0.0, min(180.0, p10))
            self._kit.servo[SERVO_CH_11].angle = max(0.0, min(180.0, p11))
            time.sleep(delay)

    def _move_servo_ramp(self, channel, start_angle, target_angle, duration=0.5):
        """Cosine-eased ramp of a single servo channel."""
        if self._kit is None:
            return
        steps = SERVO_RAMP_STEPS
        delay = duration / steps
        delta = target_angle - start_angle
        for i in range(steps + 1):
            t = i / steps
            ramp = (1.0 - math.cos(t * math.pi)) / 2.0
            ang = start_angle + delta * ramp
            self._kit.servo[channel].angle = max(0.0, min(180.0, ang))
            time.sleep(delay)

    def _independent_ch10_worker(self, fallback_start_angle):
        """Ch 10 timeline detached from the main loop while M2 runs."""
        print(f"[Servo] M2 running. Ch10 cushioning {POST_PAUSE_DELAY}s...")
        if self._kit and self._kit.servo[SERVO_CH_10].angle is not None:
            cur = self._kit.servo[SERVO_CH_10].angle
        else:
            cur = fallback_start_angle

        target = cur - CH10_SWEEP_DROP
        self._move_servo_ramp(SERVO_CH_10, cur, target, duration=0.1)

        print(f"[Servo] Ch10 sweep complete. Holding {SWEEP_HOLD_S}s...")
        time.sleep(SWEEP_HOLD_S)

        print("[Servo] Returning Ch10 to standby.")
        self._move_servos_smooth(STANDBY_10, STANDBY_11, duration=0.6)

    # ── Motor helpers ──────────────────────────────────────────────────────

    def _coils_off(self):
        """De-energise Motor 2 to prevent overheating."""
        for p in self._m2_pins:
            p.off()

    def _apply_step(self, step):
        for pin, val in zip(self._m2_pins, step):
            pin.on() if val else pin.off()

    def _send_pulses(self, num_pulses: int):
        """Send step pulses to Motor 1."""
        for _ in range(num_pulses):
            if not self._is_running:
                break
            self._pul.on()
            time.sleep(0.00004)
            self._pul.off()
            time.sleep(0.005)
            self._pulse_count += 1

    def _run_motor2_until_switch(self, reject: bool):
        """Drive Motor 2 until the limit switch is held for SWITCH_SUSTAIN_TIME."""
        start_time       = time.monotonic()
        idx              = 0
        first_closed     = None

        action_label = "REJECTING" if reject else "ACCEPTING"
        self._status_cb(f"Status: Cycle {self._cycle_count} – M2 {action_label} "
                        f"(awaiting limit switch)...")

        while self._is_running:
            elapsed = time.monotonic() - start_time

            if elapsed >= SWITCH_LOCKOUT_TIME:
                if self._limit_sw.is_pressed:
                    if first_closed is None:
                        first_closed = time.monotonic()
                        self._status_cb(f"Status: Cycle {self._cycle_count} – "
                                        f"Switch closed, verifying...")
                    elif time.monotonic() - first_closed >= SWITCH_SUSTAIN_TIME:
                        self._status_cb(f"Status: Cycle {self._cycle_count} – "
                                        f"M2 stopped by limit switch")
                        break
                else:
                    if first_closed is not None:
                        first_closed = None
                        self._status_cb(f"Status: Cycle {self._cycle_count} – "
                                        f"M2 running (switch signal lost)")

            self._apply_step(self._sequence[idx % 8])
            idx += 1
            time.sleep(self._step_delay)

    def _auto_cycle(self):
        """Main loop: M1 advances → M2 ejects/accepts → 5 s pause → repeat,
        with servos synchronised to each phase."""
        while self._is_running:
            # ── Fetch the pending sort decision (if any) ──────────────────
            try:
                decision = self._result_queue.get_nowait()
                self.last_decision = decision
            except queue.Empty:
                decision = SortDecision("unknown", reject=False)

            reject = decision.reject

            # ── Phase 1 / MOVING: M1 advances, servos stay in HOLD ────────
            self.phase = "MOVING"
            self.inference_allowed.clear()        # AI OFF while moving
            self._status_cb(
                f"Status: Cycle {self._cycle_count + 1} – MOVING "
                f"({'REJECT' if reject else 'ACCEPT'}) | servos holding | AI off..."
            )
            self._coils_off()
            self._send_pulses(PULSES_PER_CYCLE)
            if not self._is_running:
                break

            self._cycle_count += 1

            # ── Phase 2: M2 runs → servo recovery (Ch11 home, Ch10 sweep) ─
            self._status_cb(f"Status: Cycle {self._cycle_count} – "
                            f"M2 running (servo recovery triggered)...")
            print("[Servo] M2 starting. Returning Ch11 to standby immediately.")
            if self._kit and self._kit.servo[SERVO_CH_10].angle is not None:
                curr_10 = self._kit.servo[SERVO_CH_10].angle
            else:
                curr_10 = STANDBY_10 - HOLD_OFFSET

            # Immediate: Ch 11 back to home base
            self._move_servos_smooth(curr_10, STANDBY_11, duration=0.6)
            # Parallel: independent Ch 10 sweep alongside M2
            threading.Thread(target=self._independent_ch10_worker,
                             args=(curr_10,), daemon=True).start()

            # Drive M2 until the limit switch stops it
            self._run_motor2_until_switch(reject)
            self._coils_off()
            if not self._is_running:
                break

            # ── Phase 3 / PAUSING: servos snap DOWN, AI enabled ───────────
            self.phase = "PAUSING"
            self._status_cb(f"Status: Cycle {self._cycle_count} – "
                            f"M2 paused. Engaging servos to hold...")
            print("[Servo] M2 paused. Snapping servos into hold posture.")
            self._move_servos_smooth(STANDBY_10 - HOLD_OFFSET,
                                     STANDBY_11 - HOLD_OFFSET, duration=0.8)

            self.inference_allowed.set()          # AI ON for the pause
            print("[MOTOR] Inference window OPEN (PAUSING)")
            self._status_cb(
                f"Status: Cycle {self._cycle_count} complete – "
                f"PAUSING {CYCLE_PAUSE_TIME}s (AI active, servos holding)..."
            )

            # Reject paddle: fire Ch 6 if a CAM 1-rejected pepper has reached
            # the paddle station this cycle (decided by the camera system's FIFO).
            if self._paddle_check is not None:
                try:
                    if self._paddle_check():
                        print("[Servo] Reject paddle due → firing Ch6 this pause.")
                        self._reject_paddle_sweep()
                except Exception as e:
                    print(f"[Servo] Paddle check error: {e}")

            # Bin gates: fire the gate servo for each bin that a pepper reached
            # this cycle (decided by the camera system's drop FIFO).
            if self._bin_gate_check is not None:
                try:
                    for bin_label in self._bin_gate_check():
                        self.fire_bin_gate(bin_label)
                except Exception as e:
                    print(f"[Servo] Bin gate check error: {e}")

            for _ in range(int(CYCLE_PAUSE_TIME * 10)):
                if not self._is_running:
                    break
                time.sleep(0.1)

            self.inference_allowed.clear()        # AI OFF again
            print("[MOTOR] Inference window CLOSED")
            # Servos remain down as the loop restarts Phase 1 (M1 moving)

        # Loop exited
        self.phase = "IDLE"
        self.inference_allowed.clear()
        self._servos_to_standby()
