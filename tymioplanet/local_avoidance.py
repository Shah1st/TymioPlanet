"""Local navigation: wall following around obstacles that are not on the map.

Two halves:
  - THYMIO_PROGRAM runs on the robot (sent with tdmclient, which translates this Python
    subset to Aseba). It is a state machine driven by the five front proximity sensors:
    0 follow the path, 1 turn away, 2 follow the wall, 3 wait, 4 blocked. In state 0 it
    applies following_left_speed / following_right_speed, which the computer writes at
    every control step.
  - The functions below run on the computer. They remember where on the planned path the
    obstacle was met, so that the end of the avoidance can be decided from the progress
    made along that path.

Lengths here are in the unit of the path they are given (millimetres in this project).
"""

import math

# Hooks set by the main loop before the robot starts.
get_robot_pose = None  # function returning (x, y, theta) or None: use the filter estimate
optimal_path = []      # the planned waypoints, [(x, y), ...]

# State of the avoidance, shared by the functions below.
avoidance_active = False
rejoin_requested = False
exit_confirmations = 0
hit_progress = 0.0
avoidance_path = []


def closest_point_on_path(robot_x, robot_y, path):
    
    #initialisation des variables
    min_distance = float("inf") #smallest distance found between the robot and its path
    best_progress = 0.0 #représente la position du point le plus proche le long du chemin optimal, mesurée depuis le début de ce chemin = AQ ; à renommer en "distance_along_path". 
                        #C'est la distance cumulée le long du chemin optimal
    
    
    best_point = None #Stores the point that is the nearest to the optimal trajectory
    best_segment_index = -1 #indice i du waypoint de départ du segment le plus proche

    cumulative_length = 0.0 #Cumulated length on the optimal path

    for i in range(len(path) - 1):

        ax, ay = path[i]
        bx, by = path[i + 1]

        dx = bx - ax
        dy = by - ay

        segment_length = math.hypot(dx, dy) #computes sqrt(dx²+dy²) / Distance of AB

        if segment_length == 0:
            continue

        # Projection onto the segment
        t = ((robot_x - ax) * dx + (robot_y - ay) * dy) / (segment_length ** 2)
        t = max(0.0, min(1.0, t)) #computation of t 

        # Closest point
        qx = ax + t * dx
        qy = ay + t * dy
        distance = math.hypot(robot_x - qx,robot_y - qy)

        if distance < min_distance:
            min_distance = distance
            best_progress = cumulative_length + t * segment_length #Calcule le meileur progrès de chemin effectué pour le moment
            best_point = (qx, qy)
            best_segment_index = i

        cumulative_length += segment_length

    return min_distance, best_progress, best_point, best_segment_index



#===========================================================================================
#===========================================================================================
#===========================================================================================
#STARTING AVOIDANCE
#===========================================================================================
#===========================================================================================
#===========================================================================================


def start_avoidance(x, y, optimal_path):

    global avoidance_active, rejoin_requested
    global hit_progress, avoidance_path
    global exit_confirmations

    # Save (copy) the current optimal path
    avoidance_path = list(optimal_path)

    # Find where the robot encountered the obstacle
    distance, progress, point, segment_index = closest_point_on_path(x, y, avoidance_path)
    #distance: distance entre la position du robot et le point le plus proche sur le chemin optimal
    #progress: distance parcourue le long du chemin optimal, depuis son départ jusqu'au point du chemin le plus proche du robot
    #point: coordonnées du point du chemin optimal le plus proche du robot
    #segment: indice du segment sur lequel se trouve ce point

    if segment_index == -1: #aucun segment utilisable n'a été trouvé
        avoidance_active = False #alors le notebook n'active pas la surveillance automatique de la fin du contournement
        return False #et la fonction s'arrête immédiatement
        #le robot peut continuer son évitement, mais dans l'état il ne pourra pas déterminer correctement le retour auc hemin avec les infos qu'il possède acutellement

    # Remember progression at obstacle encounter
    hit_progress = progress #crucial: permet de voir le delta de progression entre les deux parties de l'obstacle

    # Initialize avoidance state
    avoidance_active = True #avoidance just started
    rejoin_requested = False #notebook didnt ask thymio to go back to path following yet
    exit_confirmations = 0 #reset exit confirmation

    return True


#===========================================================================================
#===========================================================================================
#==================ON_EVENT_DATA==============================================================
#===========================================================================================
#===========================================================================================

def on_event_data(node, event_name): #this function is called every single time an "emit()" function is called.
                                     #in this case, function with arguments are passed automatically. 
    global avoidance_active, rejoin_requested, exit_confirmations
    #avoidance active: indique si le notebook surveille déjà actuellement un contournement 
    #rejoin requested: indique si le notebook a déjà demandé au thymio de terminer son contournement
    #exit confirmations: C'est le compteur utilisé par check_avoidance_exit()
    from tdmclient.notebook import clear_event_data, get_event_data  # needs the robot link
    events = get_event_data(event_name) #qd on emit("a") avec thymio, event_name = "a" 
   
    for event in events:
        # Obstacle encountered
        # On trouve la position du robot actuelle, on l'assigne à la variable pose si celle-ci n'est pas nulle
        # Si c'est pas fiable, ça dégage
        if event_name == "avoidance_started":
            # Latest pose from the localisation module
            pose = get_robot_pose() #with camera lol, to be done

            if pose is not None:
                x, y, theta = pose
                start_avoidance(x, y, optimal_path)
            else:
                # No reliable starting pose:
                # do not authorise automatic rejoining
                # + exit the program
                avoidance_active = False
                print("No valid pose at avoidance start")
                exit(0)

        # Thymio has stopped wall following
        elif event_name == "avoidance_finished":
            avoidance_active = False
            rejoin_requested = False
            exit_confirmations = 0
            print("Avoidance finished, path following resumed")

        # Avoidance failed
        elif event_name == "avoidance_blocked":
            avoidance_active = False
            rejoin_requested = False
            exit_confirmations = 0
            print("Robot blocked")

    clear_event_data(event_name)


# The program below is not run here: it is sent to the Thymio.
THYMIO_PROGRAM = r'''
speed0 = 100       # nominal speed
obstThrL = 10      # low obstacle threshold to switch state 1->0
obstThrH = 20      # high obstacle threshold to switch state 0->1

turn_speed = 50

state = 0          # 0 = following path, 1 = initial rotation, 2 wall following, 3 waiting for path following, 4 blocked

direction = 0           # -1 = right, follow left wall, +1 = left, vice-versa
obst = [0, 0, 0, 0, 0]

timer_period[0] = 10


# Set from the computer at every control step (path follower output, Thymio units):
following_left_speed = 0
following_right_speed = 0
# Not used yet: 1 while the camera sees the field (Observation.camera_ok), 0 when hidden.
camera_state = 1

#wall-following parameters
wall_speed = 60
wall_correction = 20

#Mapping event frequency
map_counter = 0

# Will be replaced by the real exit condition
obstacle_avoided = 0

# Replanning completed by notebook
path_ready = 0


# CORNER RECOVERY PARAMETERS
# 0 = normal wall following
# 1 = move forward half a robot length
# 2 = rotate approximately 45 degrees
# 3 = move forward another half length
# 4 = rotate until the wall is detected again

wall_recovery = 0
wall_counter = 0
wall_seen = 0

# Values to calibrate experimentally
half_length_cycles = 30
turn_45_cycles = 20
search_limit = 150

clear_corner_speed = 25
corner_turn_speed = 35


@onevent
def timer0():
    global prox_horizontal, motor_left_target, motor_right_target, state, direction, obst, map_counter, wall_recovery, wall_counter, wall_seen
    
#if camera_state == 0: #If camera is inactive of senses very low shades

#Then it will start to read the sensors

    for i in range(5): #measuring every sensors values
        obst[i] = prox_horizontal[i]

    # Maximal intensity to chose if we rotate left or right
    left = obst[0]
    if obst[1] > left:
        left = obst[1]

    right = obst[3]
    if obst[4] > right:
        right = obst[4]

    # Now we check if something is in front of the Thymio
    avant_max = obst[2]
    if obst[1] > avant_max:
        avant_max = obst[1]
    if obst[3] > avant_max:
        avant_max = obst[3]

    #Maximal intensity among all sensors
    obst_max = avant_max
    if obst[0] > obst_max:
        obst_max = obst[0]
    if obst[4] > obst_max:
        obst_max = obst[4]


#checking if this is a corridor 

    # If both sides are occupied, but it is free in front of us, then it is a corridor (only the 4th and 0th sensors are concerned)
    corridor = 0
    if avant_max < obstThrL:
        if obst[0] > obstThrH:
            if obst[4] > obstThrH:
                corridor = 1

    #TODO: How to cross a corridor, how to check if the width works ? 


#We send to the notebook 

    map_counter += 1

    if map_counter >= 4: #This counter allows not to emit too much informations
        map_counter = 0

# Here, Finite State Machine (#FSM lol)

    if state == 0: #If robot follows the path
        if obst_max > obstThrH:
            state = 1 #Now it is in obstacle avoidance mode
            # Initial rotation direction
            direction = -1

            if right > left:
                direction = 1
            # Reset corner recovery
            wall_recovery = 0
            wall_counter = 0
            wall_seen = 0
            emit("avoidance_started")

    elif state == 1:
        # Switch to wall following when the front becomes practicable
        if avant_max < obstThrL:
            state = 2


    # State 2: Wall Following, exit by rejoin_path()
    # State 3: Waiting, exit by resume_path()
    # State 4: Blocked


#Control of motors to make it work

    if state == 0:
        # Normal path following
        motor_left_target = following_left_speed
        motor_right_target = following_right_speed

    elif state == 1:
        # Initial rotation
        motor_left_target = -direction * turn_speed
        motor_right_target = direction * turn_speed

    elif state == 2:
        # Wall following part handled here
        # Here, we determine the followed wall
        if direction == -1:
            wall_sensor = obst[0]
            wall_side = 1
        else:
            wall_sensor = obst[4]
            wall_side = -1

        # Check if a front obstacle requires an avoidance turn
        front_obstacle = 0

        if avant_max > obstThrH:

            if wall_recovery == 0:
                front_obstacle = 1

            elif wall_recovery == 1:
                front_obstacle = 1

            elif wall_recovery == 3:
                front_obstacle = 1


        # Front obstacle: priority over forward movements
        if front_obstacle == 1:
            wall_recovery = 0
            wall_counter = 0

            motor_left_target = -direction * turn_speed
            motor_right_target = direction * turn_speed

        # Normal wall following
        elif wall_recovery == 0:

            if wall_sensor > obstThrL: #enregistrer que le robot a déjà detecté un mur
                wall_seen = 1

            if wall_sensor > obstThrH: #correction de suivi de mur
                motor_left_target = wall_speed + wall_side * wall_correction
                motor_right_target = wall_speed - wall_side * wall_correction

            elif wall_sensor < obstThrL:
                if wall_seen == 1: #si le robot suivait un mur avant, et que 
                                    #maintenant il ne le voit plus, il commence
                                    #à pivoter pour le retrouver 
                    # Wall lost: start corner recovery
                    wall_recovery = 1
                    wall_counter = 0

                    motor_left_target = clear_corner_speed
                    motor_right_target = clear_corner_speed

                else: #Important de le laisser, car signifie que le robot 
                      #ne détècte pas le mur actuellement mais il n'en a
                      #encore jamais détecté ; alors il avance tout droit
                      #sans déclencher la récupération d'un coin.
                    motor_left_target = wall_speed
                    motor_right_target = wall_speed

            else:
                motor_left_target = wall_speed
                motor_right_target = wall_speed


# PHASE 1: FORWARD HALF LENGTH
        elif wall_recovery == 1:
            wall_counter += 1

            motor_left_target = clear_corner_speed
            motor_right_target = clear_corner_speed
            #La ligne demande de savoir combien de cycles sont effectués
            #pour atteindre une demi distance en dépit de la vitesse et de
            #la fréquence d'appel de la fonction.
            if wall_counter >= half_length_cycles:
                wall_recovery = 2 #on repasse en mode rotation de 45°
                wall_counter = 0 #on reset le counter

# PHASE 2: TURN 45 DEGREES TOWARDS WALL
        elif wall_recovery == 2:
            wall_counter += 1

            motor_left_target = -wall_side * corner_turn_speed
            motor_right_target = wall_side * corner_turn_speed

            #La ligne demande de savoir combien de cycles sont effectués
            #pour atteindre 45° en dépit de la vitesse de rotation et de
            #la fréquence d'appel de la fonction.
            if wall_counter >= turn_45_cycles:
                wall_recovery = 3 #2e avance de demi longueur
                wall_counter = 0

# PHASE 3: FORWARD HALF LENGTH
        elif wall_recovery == 3:
            wall_counter += 1

            motor_left_target = clear_corner_speed
            motor_right_target = clear_corner_speed

            if wall_counter >= half_length_cycles:
                wall_recovery = 4 #ultime mode: rotation jusqu'à retrouver le mur
                wall_counter = 0

 # PHASE 4: ROTATE UNTIL WALL IS FOUND
        elif wall_recovery == 4:
            wall_counter += 1

            if wall_sensor > obstThrH: #peut-être mettre un treshold plus grand pour les angles aigus

                wall_recovery = 0
                wall_counter = 0

                motor_left_target = 0
                motor_right_target = 0

            elif wall_counter >= search_limit: #rien n'a été trouvé => on a plus de wall following, état blocké

                state = 4

                motor_left_target = 0
                motor_right_target = 0

                emit("avoidance_blocked")

            else:

                motor_left_target = -wall_side * corner_turn_speed
                motor_right_target = wall_side * corner_turn_speed

    else:
        # State 3 or 4
        motor_left_target = 0
        motor_right_target = 0
'''
