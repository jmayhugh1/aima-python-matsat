import argparse
import logging
import sys
import time
import json
import os

import pandas as pd
from geopy.distance import geodesic

import searobotics_surveyor.surveyor_lib.helpers as hlp
import searobotics_surveyor.surveyor_lib.surveyor as surveyor


def start_mission(boat, count=5):
    """
    Start the mission by waiting for the operator to switch to waypoint mode.

    Args:
        boat (Surveyor): The Surveyor object representing the boat.
    """
    boat.set_standby_mode()
    countdown(count, "Starting mission in", "!")
    print("Mission started!")


def is_clear(boat):
    """
    Check if the path is clear for the boat to navigate.

    Args:
        boat (Surveyor): The Surveyor object representing the boat.

    Returns:
        bool: True if the path is clear, False otherwise.
    """
    return True


def avoid_obstacle(boat):
    """
    Perform actions to avoid an obstacle in the boat's path.

    Args:
        boat (Surveyor): The Surveyor object representing the boat.
    """
    boat.set_standby_mode()
    print("Obstacle detected, waiting for 10 seconds to see if it moves.")
    time.sleep(10)
    if not is_clear(boat):
        boat.set_erp_mode()
        print("Obstacle not cleared, aborting mission.")
        sys.exit(1)
    print("Obstacle avoided, resuming navigation")
    boat.set_waypoint_mode()


def countdown(count, message, additional_message=""):
    """
    Print a countdown with the given message and optional additional message.

    Args:
        count (int): The number of seconds to count down.
        message (str): The message to display before the countdown.
        additional_message (str, optional): An additional message to display after the countdown.
    """
    for i in range(count, 0, -1):
        print(f"{message} {i} {additional_message}", end="\r")
        time.sleep(1)
    print()


def load_waypoints(target):
    """
    Load waypoints from a file or folder.

    Args:
        target (str): Path to a folder or file.

    Returns:
        list: List of (lat, lon) tuples.
    """
    if os.path.isdir(target):
        # Look for waypoints.json first, then waypoints.csv
        wp_json = os.path.join(target, "waypoints.json")
        wp_csv = os.path.join(target, "waypoints.csv")
        if os.path.exists(wp_json):
            with open(wp_json, 'r') as f:
                return [tuple(wp) for wp in json.load(f)]
        elif os.path.exists(wp_csv):
            return hlp.read_csv_into_tuples(wp_csv)
        else:
            raise FileNotFoundError(f"No waypoints.json or waypoints.csv found in {target}")

    if target.endswith('.json'):
        with open(target, 'r') as f:
            return [tuple(wp) for wp in json.load(f)]
    elif target.endswith('.csv'):
        return hlp.read_csv_into_tuples(target)
    else:
        # Default to CSV behavior for backward compatibility if no extension
        return hlp.read_csv_into_tuples(target)


def load_erp(target, erp_filename=None):
    """
    Load ERP from a file, folder, or explicitly provided path.

    Args:
        target (str): Path to the target folder or file.
        erp_filename (str, optional): Explicit path to ERP file.

    Returns:
        tuple: (lat, lon) tuple.
    """
    if erp_filename:
        if erp_filename.endswith('.json'):
            with open(erp_filename, 'r') as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    return tuple(data[0]) if isinstance(data[0], list) else tuple(data)
                return tuple(data)
        return hlp.read_csv_into_tuples(erp_filename)[0]

    if os.path.isdir(target):
        for ext in ['json', 'csv']:
            erp_path = os.path.join(target, f"erp.{ext}")
            if os.path.exists(erp_path):
                if ext == 'json':
                    with open(erp_path, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list) and len(data) > 0:
                            return tuple(data[0]) if isinstance(data[0], list) else tuple(data)
                        return tuple(data)
                return hlp.read_csv_into_tuples(erp_path)[0]

    # If no ERP found and target is a file, we might not have an ERP
    return None


def main(target, erp_filename=None, mission_postfix="", mock=False):
    """
    Main function to execute the surveyor boat mission.

    Args:
        target (str): The path to the file or folder containing waypoints.
        erp_filename (str): Optional path to the file containing ERP data.
        mission_postfix (str): The suffix to be appended to the name if the CSV file containing the data.
        mock (bool): Whether to use MockSurveyor instead of real hardware.
    """
    print(f"Loading mission from {target}")
    waypoints = load_waypoints(target)
    erp = load_erp(target, erp_filename)

    if not erp:
        print("Warning: No ERP found. Using first waypoint as ERP.")
        erp = waypoints[0]

    print(f"Loaded {len(waypoints)} waypoints. ERP: {erp}")

    THROTTLE = 25  # Default throttle value
    index = 0  # Initialization variables
    ONLY_AT_WAYPOINT = False  # Set it to true if you want a separate data csv file collected ONLY at the waypoints
    data_to_be_collected = ["state", "exo2"]

    if mock:
        from distributed.mock_surveyor import LocalMockSurveyor
        boat_class = LocalMockSurveyor
    else:
        boat_class = surveyor.Surveyor

    boat_args = {
        "sensors_to_use": ["exo2", "camera"],
        "sensors_config": {
            "exo2": {"server_ip": "192.168.0.20"},
            "camera": {},
            "lidar": {},
        },
        "logger_level": logging.INFO,
    }
    if mock:
        boat_args["initial_coords"] = waypoints[0]

    boat = boat_class(**boat_args)
    with boat:
        start_mission(boat, 1)
        print(
            pd.DataFrame([boat.get_data(data_to_be_collected)])
        )  # Show example of data being collected
        current_coordinates = boat.get_gps_coordinates()

        while index < len(waypoints):
            print(f"Loading waypoint #{index + 1}")
            desired_coordinates = waypoints[index]
            boat.go_to_waypoint(desired_coordinates, erp, THROTTLE)

            while (control_mode := boat.get_control_mode()) != "Station Keep":
                if control_mode == "Go To ERP":
                    print("Going to ERP, aborting mission")
                    sys.exit(1)
                elif control_mode != "Waypoint" and not (mock and control_mode == "Waypoint"):
                    # For real boat, we wait for Waypoint mode.
                    # For mock, LocalMockSurveyor transitions to Waypoint immediately.
                    print(f"{control_mode} mode")
                    if not mock:
                        continue

                print(f"Navigating to waypoint #{index + 1}")

                if not is_clear(boat):
                    avoid_obstacle(boat)

                data = hlp.process_gga_and_save_data(
                    boat, data_keys=data_to_be_collected, post_fix=mission_postfix
                )  # You may save and retreive data at the same time

                current_coordinates = boat.get_gps_coordinates()
                print(
                    f"Meters to next waypoint {geodesic(current_coordinates, desired_coordinates).meters:.2f}"
                )
                
                if mock:
                    # In mock mode, we simulate reaching the waypoint and transitioning to Station Keep
                    print("[Mock] Waypoint reached, switching to Station Keep...")
                    boat.set_station_keep_mode()
                    break

            if mock or hlp.are_coordinates_close(
                boat.get_gps_coordinates(), desired_coordinates, tolerance_meters=2.5
            ):
                print("Successful waypoint")
                if ONLY_AT_WAYPOINT:
                    hlp.process_gga_and_save_data(
                        boat,
                        data_keys=data_to_be_collected,
                        post_fix=mission_postfix + "only_waypoints",
                    )
                index += 1
            else:
                print("Failed waypoint")
                # TODO: Implement maximum number of retries.

    print("Mission completed.")
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Waypoint mission script.")

    # Add required positional argument
    parser.add_argument("target", type=str, help="The filename or folder for waypoints.")

    # Add optional ERP filename
    parser.add_argument("erp_filename", type=str, nargs="?", default=None, help="Optional filename for ERP.")

    # Add optional argument with a default value
    parser.add_argument(
        "mission_postfix",
        type=str,
        nargs="?",
        default="",
        help="Optional mission postfix.",
    )

    # Add mock flag
    parser.add_argument("--mock", action="store_true", help="Use MockSurveyor instead of real hardware.")

    # Parse the command line arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args.target, args.erp_filename, args.mission_postfix, args.mock)