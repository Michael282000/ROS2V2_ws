[README (4).md](https://github.com/user-attachments/files/34965747/README.4.md)
# ROS2V2_ws

ROS 2 workspace voor het aansturen van een **Doosan cobot (model A0509)** via de `dsr_controller2` van de Doosan ROS 2 driver. Het bevat losse Python-scripts om de robot te laten bewegen, een kleine interpreter voor een JSON-taal (RTSL v0.1) en een kalibratiescript voor de IK-oplossingsruimte.

## Waar staat de code?

Alle code zit in één ament_python-package:

```
ROS2V2_ws/
└── src/
    └── mijn_eerste_pkg/
        ├── package.xml / setup.py / setup.cfg   # package-configuratie (nog TODO-velden)
        ├── mijn_eerste_pkg/__init__.py          # lege Python-module
        ├── resource/mijn_eerste_pkg             # ament resource-marker
        └── test/                                # <-- hier staan de scripts
            ├── Robot_bewegen.py
            ├── Robot_Bewegen_Robot_Code.py
            ├── rtsl_doosan_executor.py
            ├── rtsl_doosan_executor_2.py
            ├── sol_test.py
            ├── Human.json
            └── test_copyright.py / test_flake8.py / test_pep257.py   # standaard ament-lint tests
```

De scripts staan in `src/mijn_eerste_pkg/test/` en zijn **geen** ROS-nodes met `entry_points`; je draait ze direct met `python3`.

Overige bestanden in de root: `frames_2026-09-28_15.38.18.gv/.pdf` (TF-boom van `view_frames`: `world → base_link`) en `.gitignore` (negeert `build/`, `install/`, `log/`).

## Scripts

| Script | Doel |
|---|---|
| `Robot_bewegen.py` | Leest de huidige jointhoeken (`GetCurrentPosj`) en beweegt veilig naar één absolute joint-target (`MoveJoint`). Standaard `ENABLE_MOTION = False`: alleen uitlezen en tonen. |
| `Robot_Bewegen_Robot_Code.py` | Voert een JSON-programma (`Human.json`) uit: `set`, `ptp`, `move-ee` (PTP) en `wait`. Het hele script wordt vóór de eerste beweging gecontroleerd (preflight). |
| `rtsl_doosan_executor.py` | RTSL v0.1 interpreter, uitgebreid met `move-ee mode=lin` (rechte lijn, SLERP voor oriëntatie, branch-controle via IK/FK-sampling). |
| `rtsl_doosan_executor_2.py` | Nieuwere revisie van de executor: poses in `DR_WORLD`, PTP voert de gevalideerde IK-oplossing uit via `MoveJoint`, LIN-preflight vanaf de gemeten starttoestand, LIN wordt geweigerd tenzij `DOOSAN_MOVELINE_ORIENTATION_IS_SLERP` bewust is gezet. |
| `sol_test.py` | Kalibreert `CANONICAL_CONFIG_TO_DOOSAN_SOL` (mapping SL/SR, ED/EU, WN/WF → Doosan `sol` 0..7). **Beweegt de robot niet**; gebruikt alleen `Ikin`, `Fkin` en `GetCurrentPosj`. |
| `Human.json` | Voorbeeldprogramma in RTSL-formaat. |

## JSON-taal (RTSL v0.1)

Ondersteunde taken:

```json
{"task": "set", "param": "velocity", "value": 0.5}
{"task": "ptp", "value": [0, -30, 60, 0, 60, 0], "vel": 0.5}
{"task": "move-ee", "mode": "ptp",
 "pose": {"pos": [0.40, -0.10, 0.25], "quat": [0.0, 0.7071, 0.7071, 0.0]},
 "config": {"shoulder": "SR", "elbow": "EU", "wrist": "WN"}, "vel": 0.25}
{"task": "move-ee", "mode": "lin",
 "pose": {"pos": [0.40, -0.10, 0.25], "quat": [0.0, 0.7071, 0.7071, 0.0]}, "vel": 0.05}
{"task": "wait", "value": 2}
```

- Posities in **meter**, oriëntatie als quaternion `(w, x, y, z)`, wereld → TCP.
- Doosan werkt intern met mm en ZYZ-Euler in graden; de scripts rekenen dit om.
- PTP `vel`/`acc` zijn fracties (0..1) van de PTP-maxima; LIN `vel`/`acc` zijn TCP m/s en m/s².
- Configuratie → Doosan-oplossing: `sol = 4·S + 2·E + W` (S: SL=0/SR=1, E: ED=0/EU=1, W: WN=0/WF=1).

## Vereisten

- Ubuntu met ROS 2 (de `__pycache__` wijst op Python 3.12, dus waarschijnlijk ROS 2 Jazzy)
- Doosan ROS 2 driver met het `dsr_msgs2`-pakket (services `GetCurrentPosj`, `Ikin`, `Fkin`, `MoveJoint`, `MoveJointx`, `MoveLine`)
- `rclpy`, `colcon`

## Gebruik

```bash
# 1. Bouwen
cd ~/ROS2V2_ws
colcon build --packages-select mijn_eerste_pkg
source install/setup.bash

# 2. Start de Doosan bringup (virtual of real) in een andere terminal
#    De scripts verwachten robot-ID "dsr01" en de namespace /dsr01/dsr_controller2

# 3. Scripts draaien
python3 src/mijn_eerste_pkg/test/sol_test.py                 # kalibratie, geen beweging
python3 src/mijn_eerste_pkg/test/Robot_bewegen.py            # uitlezen / joint-move
python3 src/mijn_eerste_pkg/test/rtsl_doosan_executor_2.py   # Human.json uitvoeren
```

## Instellingen aanpassen

Bovenin elk script staat een blok `USER SETTINGS`:

- `ROBOT_ID` – standaard `dsr01`
- `ENABLE_MOTION` – `False` = alleen lezen en valideren, `True` = daadwerkelijk bewegen
- `INSTRUCTION_FILE` – pad naar het JSON-programma. **Dit is nu hardcoded** naar `/home/bbrug1/ROS2V2_ws/src/mijn_eerste_pkg/test/Human.json`; pas het aan naar jouw eigen pad
- Snelheids- en versnellingslimieten (`DOOSAN_PTP_MAX_*`, `BACKEND_LIN_MAX_*`) en `JOINT_LIMITS_DEG`

## Veiligheid

- Test altijd eerst op de **virtual** Doosan bringup voordat je de echte robot gebruikt.
- In `Robot_Bewegen_Robot_Code.py` en beide `rtsl_doosan_executor*.py` staat `ENABLE_MOTION = True`: de robot beweegt dan echt. Zet dit eerst op `False` voor een dry-run.
- Controleer de limieten en controller-specifieke waarden (maximumsnelheden, SLERP-gedrag van `MoveLine`) op de fysieke opstelling.
- Houd de noodstop binnen handbereik.

## Status

Werk in uitvoering: `package.xml` en `setup.py` bevatten nog `TODO`-velden (beschrijving, licentie) en er zijn nog geen `console_scripts`.
