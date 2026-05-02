#!/usr/bin/env python3
"""
sensor_publisher.py  —  v4.0  (version hospitalière définitive)
===============================================================
Nœud ROS 2 — Simulateur de capteurs électriques pour hôpital camerounais ~100 lits.

Référence :
  Moukengué Imano A., "Problèmes d'électrification urbaine au Cameroun",
  Université de Douala, Novembre 2015.

Réseau simulé  : BTA 220 V / 50 Hz (art. 14 Règlement MINEE)
Environnement  : Hôpital ~100 lits, Cameroun
Cycle          : toutes les 2 secondes

Objectif ML :
  La variable cible est "outage" (coupure complète).
  Le modèle doit la prédire AVANT qu'elle ne survienne.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Bool, String
import random
import json
from datetime import datetime

# ============================================================
# CONSTANTES RÉSEAU CAMEROUNAIS (MINEE art. 14)
# ============================================================
VOLTAGE_NOMINAL   = 220.0          # V
FREQ_NOMINAL      = 50.0           # Hz

VOLTAGE_MIN_LEGAL = VOLTAGE_NOMINAL * 0.90   # 198 V
VOLTAGE_MAX_LEGAL = VOLTAGE_NOMINAL * 1.10   # 242 V
FREQ_MIN_LEGAL    = FREQ_NOMINAL    * 0.95   # 47.5 Hz
FREQ_MAX_LEGAL    = FREQ_NOMINAL    * 1.05   # 52.5 Hz

# ============================================================
# HÔPITAL ~100 LITS
# ============================================================
HOSPITAL_POWER_W = 200_000.0       # 200 kW

# ============================================================
# PROBABILITÉS RÉALISTES (coupures / microcoupures)
# ============================================================
# Objectif : ~2–4 coupures complètes PAR JOUR
P_OUTAGE = {
    'peak':  0.0003,   # 18h–22h  → ~0.7 coupure/h
    'day':   0.00015,  # 06h–18h  → ~0.3 coupure/h
    'night': 0.00005,  # 22h–06h  → ~0.1 coupure/h
}

OUTAGE_DUR_MIN = 15      # cycles (30 sec)
OUTAGE_DUR_MAX = 450     # cycles (15 min)

# Microcoupures : 3–5 par heure
P_MICRO = 0.002

# Signal précurseur AVANT chaque coupure
PRE_FAULT_MIN = 3        # cycles (6 sec)
PRE_FAULT_MAX = 8        # cycles (16 sec)


class ElectricitySensorPublisher(Node):
    """
    Machine à états (prioritaire) :
      1. outage_active      → tout à 0 (coupure complète)
      2. micro_outage_active→ tout à 0 (microcoupure)
      3. pre_fault_active   → tension chute progressivement
      4. normal             → régime normal
    """

    def __init__(self):
        super().__init__('electricity_sensor_publisher')

        # Publishers
        self.voltage_pub      = self.create_publisher(Float32, '/electricity/voltage', 10)
        self.freq_pub         = self.create_publisher(Float32, '/electricity/frequency', 10)
        self.power_pub        = self.create_publisher(Float32, '/electricity/power', 10)
        self.thd_pub          = self.create_publisher(Float32, '/electricity/thd', 10)
        self.outage_pub       = self.create_publisher(Bool,    '/electricity/outage', 10)
        self.micro_outage_pub = self.create_publisher(Bool,    '/electricity/micro_outage', 10)
        self.raw_data_pub     = self.create_publisher(String,  '/electricity/raw_data', 10)

        # États
        self.outage_active     = False
        self.outage_duration   = 0
        self.outage_counter    = 0

        self.pre_fault_active   = False
        self.pre_fault_duration = 0
        self.pre_fault_total    = 0

        self.micro_outage_active   = False
        self.micro_outage_duration = 0
        self.micro_outage_counter  = 0

        self.cycles_since_last_outage = 0
        self.cycle = 0

        # Timer 2 secondes
        self.timer = self.create_timer(2.0, self.publish_sensor_data)

        self.get_logger().info(
            f'=== HÔPITAL CAMEROUNAIS v4.0 ===\n'
            f'  Réseau : {VOLTAGE_NOMINAL} V / {FREQ_NOMINAL} Hz\n'
            f'  Puissance : {HOSPITAL_POWER_W/1000:.0f} kW\n'
            f'  Cible ML = outage (coupure complète)\n'
            f'  Précurseur : {PRE_FAULT_MIN*2}-{PRE_FAULT_MAX*2} sec avant coupure'
        )

    # --------------------------------------------------------
    # Probabilité horaire
    # --------------------------------------------------------
    def _outage_probability(self, hour: int) -> float:
        if 18 <= hour < 22:
            return P_OUTAGE['peak']
        elif 6 <= hour < 18:
            return P_OUTAGE['day']
        else:
            return P_OUTAGE['night']

    # --------------------------------------------------------
    # Machine à états
    # --------------------------------------------------------
    def simulate_outage_events(self, hour: int):
        self.cycles_since_last_outage += 1

        # Coupure en cours
        if self.outage_active:
            self.outage_duration -= 1
            if self.outage_duration <= 0:
                self.outage_active = False
                self.get_logger().info('Alimentation rétablie')
            return

        # Précurseur actif
        if self.pre_fault_active:
            self.pre_fault_duration -= 1
            if self.pre_fault_duration <= 0:
                self.pre_fault_active = False
                self.outage_active = True
                self.outage_counter += 1
                dur = random.randint(OUTAGE_DUR_MIN, OUTAGE_DUR_MAX)
                self.outage_duration = dur
                self.cycles_since_last_outage = 0
                self.get_logger().warn(
                    f'COUPURE #{self.outage_counter} — {dur*2/60:.1f} min'
                )
            return

        # Déclenchement précurseur
        if not self.micro_outage_active:
            if random.random() < self._outage_probability(hour):
                dur = random.randint(PRE_FAULT_MIN, PRE_FAULT_MAX)
                self.pre_fault_active = True
                self.pre_fault_duration = dur
                self.pre_fault_total = dur
                self.get_logger().warn(f'PRÉCURSEUR — coupure dans {dur*2} sec')
                return

        # Microcoupure
        if self.micro_outage_active:
            self.micro_outage_duration -= 1
            if self.micro_outage_duration <= 0:
                self.micro_outage_active = False
        else:
            if random.random() < P_MICRO:
                dur = random.randint(1, 5)
                self.micro_outage_active = True
                self.micro_outage_duration = dur
                self.micro_outage_counter += 1
                self.get_logger().warn(f'⚡ MICROCOUPURE #{self.micro_outage_counter}')

    # --------------------------------------------------------
    # Modèles physiques
    # --------------------------------------------------------
    def get_voltage(self, hour: int) -> float:
        if self.outage_active or self.micro_outage_active:
            return 0.0

        # Bruit réaliste (manuscrit)
        if 6 <= hour < 18:
            noise = random.gauss(-6.6, 13.0)
        else:
            noise = random.gauss(0.6, 10.0)
            if random.random() < 0.02:
                noise += random.uniform(15.0, 40.0)

        # Précurseur : chute progressive
        pre_drop = 0.0
        if self.pre_fault_active and self.pre_fault_total > 0:
            progress = 1.0 - (self.pre_fault_duration / self.pre_fault_total)
            pre_drop = -random.uniform(10.0, 50.0) * progress

        return max(0.0, round(VOLTAGE_NOMINAL + noise + pre_drop, 2))

    def get_frequency(self, hour: int) -> float:
        if self.outage_active or self.micro_outage_active:
            return 0.0

        if 6 <= hour < 18:
            base = random.gauss(51.75, 2.3)
            if random.random() < 0.03:
                base += random.uniform(8.0, 16.0)
        else:
            base = random.gauss(51.36, 1.5)
            if random.random() < 0.01:
                base += random.uniform(5.0, 17.0)
        return max(0.0, round(base, 3))

    def get_power(self, hour: int) -> float:
        if self.outage_active or self.micro_outage_active:
            return 0.0

        if 8 <= hour < 18:
            lf_max = 1.00
        elif 18 <= hour < 22:
            lf_max = 0.80
        else:
            lf_max = 0.60

        lf_min = max(0.45, lf_max - 0.15)
        lf = random.uniform(lf_min, lf_max)
        return round(HOSPITAL_POWER_W * lf + random.gauss(0, 2000), 1)

    def get_thd(self, power_w: float) -> float:
        if self.outage_active or self.micro_outage_active:
            return 0.0
        load_ratio = power_w / HOSPITAL_POWER_W
        base_thd = 8.0 + load_ratio * 7.0
        if random.random() < 0.05:
            base_thd += random.uniform(5.0, 15.0)
        return round(min(base_thd + random.gauss(0, 1.0), 40.0), 2)

    # --------------------------------------------------------
    # Publication
    # --------------------------------------------------------
    def publish_sensor_data(self):
        self.cycle += 1
        now = datetime.now()
        hour = now.hour

        self.simulate_outage_events(hour)

        voltage   = self.get_voltage(hour)
        frequency = self.get_frequency(hour)
        power     = self.get_power(hour)
        thd       = self.get_thd(power)

        # Conformité légale
        v_out = int(voltage > 0 and (voltage < VOLTAGE_MIN_LEGAL or voltage > VOLTAGE_MAX_LEGAL))
        f_out = int(frequency > 0 and (frequency < FREQ_MIN_LEGAL or frequency > FREQ_MAX_LEGAL))

        # Publication
        self.voltage_pub.publish(Float32(data=float(voltage)))
        self.freq_pub.publish(Float32(data=float(frequency)))
        self.power_pub.publish(Float32(data=float(power)))
        self.thd_pub.publish(Float32(data=float(thd)))
        self.outage_pub.publish(Bool(data=self.outage_active))
        self.micro_outage_pub.publish(Bool(data=self.micro_outage_active))

        # JSON complet
        raw = {
            "timestamp": now.isoformat(),
            "cycle": self.cycle,
            "voltage_v": voltage,
            "frequency_hz": frequency,
            "power_w": power,
            "thd_percent": thd,
            "outage": int(self.outage_active),
            "micro_outage": int(self.micro_outage_active),
            "pre_fault": int(self.pre_fault_active),
            "voltage_deviation": round(voltage - VOLTAGE_NOMINAL, 2),
            "freq_deviation": round(frequency - FREQ_NOMINAL, 3),
            "voltage_out_of_tolerance": v_out,
            "freq_out_of_tolerance": f_out,
            "hour": hour,
            "is_peak_hour": int(18 <= hour < 22),
            "is_night": int(hour < 6 or hour >= 22),
            "outage_count": self.outage_counter,
            "micro_outage_count": self.micro_outage_counter,
            "cycles_since_last_outage": self.cycles_since_last_outage,
        }

        self.raw_data_pub.publish(String(data=json.dumps(raw)))


def main(args=None):
    rclpy.init(args=args)
    node = ElectricitySensorPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('=== ARRÊT ===')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()