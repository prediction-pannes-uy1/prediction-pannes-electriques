#!/usr/bin/env python3
"""
data_collector.py
=================
Collecteur ROS 2 → fichier CSV prêt pour ML.
Aucune perte, aucune transformation illégale.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import csv
import json
import os
from datetime import datetime

VOLTAGE_NOMINAL = 220.0
FREQ_NOMINAL = 50.0
VOLTAGE_MIN_LEGAL = 198.0
VOLTAGE_MAX_LEGAL = 242.0
FREQ_MIN_LEGAL = 47.5
FREQ_MAX_LEGAL = 52.5


class DataCollectorNode(Node):
    # Ordre FIXE pour le modèle ML
    FIELDNAMES = [
        "timestamp",
        "cycle",
        "voltage_v",
        "frequency_hz",
        "power_w",
        "thd_percent",
        "outage",               # ← CIBLE PRINCIPALE
        "micro_outage",
        "pre_fault",
        "voltage_deviation",
        "freq_deviation",
        "voltage_out_of_tolerance",
        "freq_out_of_tolerance",
        "hour",
        "is_peak_hour",
        "is_night",
        "outage_count",
        "micro_outage_count",
        "cycles_since_last_outage",
    ]

    def __init__(self):
        super().__init__('data_collector')

        self.output_dir = os.path.expanduser('~/electricity_ws/data')
        os.makedirs(self.output_dir, exist_ok=True)

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_path = os.path.join(self.output_dir, f'hopital_cameroun_{ts}.csv')

        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writeheader()

        self.record_count = 0
        self.subscription = self.create_subscription(
            String,
            '/electricity/raw_data',
            self.on_data_received,
            10
        )

        self.get_logger().info(f'Collecteur actif → {self.csv_path}')

    def on_data_received(self, msg: String):
        try:
            data = json.loads(msg.data)

            # Vérification minimale
            required = ['voltage_v', 'frequency_hz', 'power_w', 'thd_percent', 'outage']
            for r in required:
                if r not in data:
                    self.get_logger().error(f'Champ manquant : {r}')
                    return

            self.record_count += 1

            with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                row = {k: data.get(k, '') for k in self.FIELDNAMES}
                writer.writerow(row)

            if self.record_count % 100 == 0:
                self.get_logger().info(f'{self.record_count} lignes enregistrées')

        except Exception as e:
            self.get_logger().error(f'Erreur : {e}')


def main(args=None):
    rclpy.init(args=args)
    node = DataCollectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info(f'Fichier final : {node.csv_path}')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()