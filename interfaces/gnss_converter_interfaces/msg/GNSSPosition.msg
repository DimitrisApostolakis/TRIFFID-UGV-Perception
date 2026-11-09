# GNSS Position message
# Contains the robot global position derived from the global TF frame

std_msgs/Header header

float64 latitude    # Latitude in degrees
float64 longitude   # Longitude in degrees
float32 heading     # Robot heading in radians, compass-style: 0 north, positive clockwise
bool gnss_valid     # True after at least one accepted RTK carrier-solution message

