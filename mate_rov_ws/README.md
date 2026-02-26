# MATE 2026 ROV — ROS 2 Workspace

## Architecture Overview

```
┌─────────────────────────────────┐          ┌──────────────────────────────────┐
│     LattePanda (Onboard)        │          │     Laptop (Control Station)     │
│     Ubuntu 22.04 + ROS 2        │          │     Ubuntu 22.04 + ROS 2        │
│                                 │   LAN    │                                  │
│  rov_onboard package:           │◄────────►│  rov_control package:            │
│    • camera_node       ───────────publish──►│    • camera_viewer_node          │
│    • sensor_node       ───────────publish──►│    • dashboard_node              │
│    • status_node       ───────────publish──►│                                  │
│    • thruster_node     ◄──────────subscribe─│    • gamepad_node                │
└─────────────────────────────────┘          └──────────────────────────────────┘
```

### ROS 2 Topics

| Topic                          | Type                          | Direction           |
|--------------------------------|-------------------------------|---------------------|
| `/rov/camera/image_raw`        | `sensor_msgs/Image`           | LattePanda → Laptop |
| `/rov/camera/image_compressed` | `sensor_msgs/CompressedImage` | LattePanda → Laptop |
| `/rov/sensor_data`             | `rov_msgs/SensorData`         | LattePanda → Laptop |
| `/rov/status`                  | `rov_msgs/RovStatus`          | LattePanda → Laptop |
| `/rov/thruster_command`        | `rov_msgs/ThrusterCommand`    | Laptop → LattePanda |

---

## File Structure

```
mate_rov_ws/
├── README.md
├── src/
│   ├── rov_msgs/                          # Shared custom messages
│   │   ├── CMakeLists.txt
│   │   ├── package.xml
│   │   └── msg/
│   │       ├── ThrusterCommand.msg
│   │       ├── SensorData.msg
│   │       └── RovStatus.msg
│   │
│   ├── rov_onboard/                       # Runs on LattePanda
│   │   ├── package.xml
│   │   ├── setup.py
│   │   ├── setup.cfg
│   │   ├── resource/rov_onboard
│   │   ├── config/
│   │   │   └── onboard_params.yaml
│   │   ├── launch/
│   │   │   ├── onboard_launch.py          # Full system launch
│   │   │   └── camera_only_launch.py      # Camera-only for testing
│   │   └── rov_onboard/
│   │       ├── __init__.py
│   │       ├── camera_node.py
│   │       ├── thruster_node.py
│   │       ├── sensor_node.py
│   │       └── status_node.py
│   │
│   └── rov_control/                       # Runs on Laptop
│       ├── package.xml
│       ├── setup.py
│       ├── setup.cfg
│       ├── resource/rov_control
│       ├── config/
│       │   └── control_params.yaml
│       ├── launch/
│       │   ├── control_launch.py          # Full control launch
│       │   └── camera_viewer_only_launch.py
│       └── rov_control/
│           ├── __init__.py
│           ├── camera_viewer_node.py
│           ├── gamepad_node.py
│           └── dashboard_node.py
```

---

## STEP-BY-STEP SETUP INSTRUCTIONS

### STEP 1: Network Configuration (BOTH machines)

Both machines must be on the same local network (e.g. connected to the same
router/switch, or a direct Ethernet cable).

**On the LattePanda**, find its IP:
```bash
ip addr show
# Look for your ethernet/wifi adapter, note the IP (e.g. 192.168.1.100)
```

**On the Laptop**, find its IP:
```bash
ip addr show
# e.g. 192.168.1.101
```

Verify they can reach each other:
```bash
# From laptop:
ping 192.168.1.100

# From LattePanda:
ping 192.168.1.101
```

### STEP 2: ROS 2 DDS Network Setup (BOTH machines)

ROS 2 uses DDS for communication. By default, nodes on the same LAN with the
same `ROS_DOMAIN_ID` will auto-discover each other.

Run this on **BOTH** machines (add to `~/.bashrc` for persistence):
```bash
echo 'export ROS_DOMAIN_ID=42' >> ~/.bashrc
source ~/.bashrc
```

> Use any number 0-232. Just make sure both machines use the **same** number.
> Using a non-zero ID avoids conflicts with other ROS 2 users on the network.

To verify DDS works across machines, on one machine run:
```bash
ros2 topic pub /test std_msgs/String "data: hello"
```
On the other machine:
```bash
ros2 topic echo /test
```
You should see `hello` appear. Press Ctrl+C to stop both.

### STEP 3: Install Dependencies (BOTH machines)

Both machines need the same workspace (so custom messages are available).
Copy this entire `mate_rov_ws` folder to **both** machines at `~/mate_rov_ws`.

You can use `scp` to copy from one machine to another:
```bash
# From the machine that has the files, copy to the other:
scp -r ~/mate_rov_ws user@192.168.1.XXX:~/mate_rov_ws
```

Or use a USB drive, git repo, etc.

**On BOTH machines**, install required system packages:
```bash
# Source ROS 2 (adjust if you use a different distro than Humble)
source /opt/ros/humble/setup.bash

# Install dependencies
sudo apt update
sudo apt install -y \
    python3-opencv \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    python3-pip

# For the gamepad node (control laptop only):
pip3 install pygame
```

### STEP 4: Build the Workspace (BOTH machines)

```bash
cd ~/mate_rov_ws

# Source ROS 2
source /opt/ros/humble/setup.bash

# Build all packages
colcon build --symlink-install

# Source the workspace
source install/setup.bash
```

> **Tip:** Add these lines to `~/.bashrc` on both machines so you don't have to
> type them every time:
> ```bash
> echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc
> echo 'source ~/mate_rov_ws/install/setup.bash' >> ~/.bashrc
> echo 'export ROS_DOMAIN_ID=42' >> ~/.bashrc
> ```

### STEP 5: Test Camera Streaming (Quick Test)

#### On the LattePanda:
```bash
cd ~/mate_rov_ws
source install/setup.bash
ros2 launch rov_onboard camera_only_launch.py
```

You should see output like:
```
[camera_node] Camera opened: 640x480 @ 30fps (index 0)
[camera_node] Published 150 frames
```

If the camera doesn't open, check:
```bash
# List available cameras:
ls /dev/video*

# If your camera is at a different index, change it:
# Edit src/rov_onboard/config/onboard_params.yaml
# Set camera_index to 2, 4, etc. then rebuild: colcon build
```

#### On the Laptop:
```bash
cd ~/mate_rov_ws
source install/setup.bash
ros2 launch rov_control camera_viewer_only_launch.py
```

An OpenCV window titled "ROV Camera Feed" should appear showing the live
camera feed from the LattePanda!

### STEP 6: Launch Full System

#### On the LattePanda (all onboard nodes):
```bash
cd ~/mate_rov_ws
source install/setup.bash
ros2 launch rov_onboard onboard_launch.py
```

#### On the Laptop (all control nodes):
```bash
cd ~/mate_rov_ws
source install/setup.bash
ros2 launch rov_control control_launch.py
```

### STEP 7: Verify Everything is Running

On either machine:
```bash
# List all active nodes:
ros2 node list

# List all active topics:
ros2 topic list

# Check camera topic publish rate:
ros2 topic hz /rov/camera/image_compressed

# View sensor data:
ros2 topic echo /rov/sensor_data

# View ROV status:
ros2 topic echo /rov/status
```

---

## TROUBLESHOOTING

### "No frames received on laptop"
1. Check both machines have `ROS_DOMAIN_ID=42` set:
   ```bash
   echo $ROS_DOMAIN_ID
   ```
2. Check firewall isn't blocking DDS multicast:
   ```bash
   sudo ufw allow in proto udp to 224.0.0.0/4
   sudo ufw allow in proto udp from 192.168.1.0/24
   ```
3. If on WiFi, some routers block multicast. Use a direct Ethernet connection
   or set up a Cyclone DDS config with unicast peers. See the "Cyclone DDS
   Unicast" section below.

### Cyclone DDS Unicast Config (for problematic networks)

Create `~/cyclonedds.xml` on **BOTH** machines:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<CycloneDDS xmlns="https://cdds.io/config">
  <Domain>
    <General>
      <Interfaces>
        <NetworkInterface autodetermine="true"/>
      </Interfaces>
    </General>
    <Discovery>
      <Peers>
        <Peer address="192.168.1.100"/>  <!-- LattePanda IP -->
        <Peer address="192.168.1.101"/>  <!-- Laptop IP -->
      </Peers>
    </Discovery>
  </Domain>
</CycloneDDS>
```

Then add to `~/.bashrc`:
```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///home/$USER/cyclonedds.xml
```

Install Cyclone DDS:
```bash
sudo apt install -y ros-humble-rmw-cyclonedds-cpp
```

### Camera index wrong
```bash
# Try different indices:
v4l2-ctl --list-devices
# or
ls /dev/video*
```

### Low framerate over network
Edit `src/rov_onboard/config/onboard_params.yaml`:
- Reduce `frame_width`/`frame_height` (e.g., 320x240)
- Lower `jpeg_quality` (e.g., 30)
- Reduce `fps` (e.g., 15)

Then rebuild: `colcon build --symlink-install`

---

## NEXT STEPS

- [ ] **Wire thrusters**: Edit `thruster_node.py` with your actual motor driver code (serial to Arduino, direct PWM, etc.)
- [ ] **Add real sensors**: Edit `sensor_node.py` to read from your depth sensor, IMU, etc.
- [ ] **Leak detection**: Add a leak sensor node to `rov_onboard`
- [ ] **Second camera**: Add another `camera_node` instance with a different `camera_index`
- [ ] **GUI dashboard**: Upgrade `dashboard_node.py` to a PyQt5/tkinter graphical display
- [ ] **Autonomous features**: Add new nodes for computer vision, PID control, etc.
