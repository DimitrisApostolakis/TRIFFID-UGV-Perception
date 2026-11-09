# Status message for Unitree B2W robot.
# Lightweight operational summary with full GNSS position.

std_msgs/Header header

# Orientation (from odometry)
float32 roll                   # Roll angle (rad)
float32 pitch                  # Pitch angle (rad)
float32 yaw                    # Heading/yaw angle (rad)
float32 heading                # Robot heading from GNSS/global TF converter (rad)

# Global position (from GNSS converter)
float64 latitude               # Latitude in degrees
float64 longitude              # Longitude in degrees
bool gnss_valid                # True after GNSS converter has seen an accepted RTK fix

# Battery and power (from LowState)
uint8 battery_percentage       # BMS state of charge (0-100); valid when has_low_state
uint8 battery_status           # Unitree BMS status, passed through without interpretation
int8 battery_temperature       # Battery temperature (C)
float32 power_voltage          # System voltage (V)
float32 power_current          # System current (A)

# System temperatures (from LowState)
int8 temperature_ntc1          # NTC temperature sensor 1
int8 temperature_ntc2          # NTC temperature sensor 2

# Robot state (from SportModeState)
uint8 mode                     # Current robot mode
uint32 error_code              # Unitree SportModeState fault code; valid when has_sport_mode_state

# Communications health (from /b2/comms_sample)
bool comms_status              # True if the base station replied
float32 value_ms               # Round-trip ping time in milliseconds

# Commander (who requested the current running action)
string commander               # e.g., "onboard", "ground_station", "none"

# Behavior (string describing current behavior from robal nav_status)
string behavior                # e.g., "IDLE", "NAVIGATING", "EMERGENCY", "FREEZE"

# Action command (JSON-serialized goal fields of the active action)
string action_command          # e.g., {"behavior": "navigation", "goal_pose": {...}}
string action_feedback         # JSON-serialized latest feedback for the active action

# Source availability: true after at least one message; these are not freshness checks.
# Summary fields default to zero/false until their source is received.
bool has_low_state
bool has_sport_mode_state
bool has_imu
bool has_odometry
bool has_global_position
gnss_converter_interfaces/GNSSPosition global_position
bool has_comms_sample

