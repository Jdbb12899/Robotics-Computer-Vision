import cozmo

from time import sleep
from cmap import *
from gui import *
from utils import *

import random

MAX_NODES = 20000

################################################################################
# NOTE:
# Before you start, please familiarize yourself with class Node in utils.py
# In this project, all nodes are Node object, each of which has its own
# coordinate and parent if necessary. You could access its coordinate by node.x
# or node[0] for the x coordinate, and node.y or node[1] for the y coordinate
################################################################################
# Edit the start x, y to change the starting point
# The angle has to point to positive x-axis direction
start_x = 130
start_y = 90

final_angle = 0
pose = (0, 0, 0)
marked = {}


def step_from_to(node0, node1, limit=75):
    ############################################################################
    # TODO: please enter your code below.
    # 1. If distance between two nodes is less than limit, return node1
    # 2. Otherwise, return a node in the direction from node0 to node1 whose
    #    distance to node0 is limit. Recall that each iteration we can move
    #    limit units at most
    # 3. Hint: please consider using np.arctan2 function to get vector angle
    # 4. Note: remember always return a Node object
    dist = get_dist(node0, node1)
    if dist < limit:
        return node1
    rad = np.arctan2(node1.y - node0.y, node1.x - node0.x)
    new_node = (node0[0] + np.cos(rad) * limit, node0[1] + np.sin(rad) * limit)
    return Node(new_node, node0)
    ############################################################################


def node_generator(cmap):
    rand_node = None
    ############################################################################
    # TODO: please enter your code below.
    # 1. Use CozMap width and height to get a uniformly distributed random node
    # 2. Use CozMap.is_inbound and CozMap.is_inside_obstacles to determine the
    #    legitimacy of the random node.
    # 3. Note: remember always return a Node object
    width, height = cmap.get_size()
    while True:
        x = random.randint(0, width)
        y = random.randint(0, height)
        parent = None
        rand_node = Node((x, y), parent)
        if cmap.is_inbound(rand_node):
            if not cmap.is_inside_obstacles(rand_node):
                break
    ############################################################################
    return rand_node


def RRT(cmap, start):
    print("RTT")
    cmap.add_node(start)

    map_width, map_height = cmap.get_size()

    while (cmap.get_num_nodes() < MAX_NODES):
        ########################################################################
        # TODO: please enter your code below.
        # 1. Use CozMap.get_random_valid_node() to get a random node. This
        #    function will internally call the node_generator above
        # 2. Get the nearest node to the random node from RRT
        # 3. Limit the distance RRT can move
        # 4. Add one path from nearest node to random node
        #
        rand_node = cmap.get_random_valid_node()
        nearest_node = None
        min_dist = 10000
        for node in cmap.get_nodes():
            if get_dist(node, rand_node) < min_dist:
                nearest_node = node
                min_dist = get_dist(rand_node, nearest_node)
        ########################################################################
        sleep(0.01)
        cmap.add_path(nearest_node, rand_node)
        if cmap.is_solved():
            break

    if cmap.is_solution_valid():
        print("A valid solution has been found :-) ")
    else:
        print("Please try again :-(")

async def turnToAngle(degree, robot: cozmo.robot.Robot):
    print("TurnToAngle")
    theta = (degree - pose[2]) % 360
    if theta >= 180:
        theta = -(360 - theta)
    await turnDegrees(theta, robot)


async def turnDegrees(degree, robot: cozmo.robot.Robot):
    print("TurnDegrees")
    global pose
    angle = (pose[2] + degree) % 360
    pose = (pose[0], pose[1], angle)
    await robot.turn_in_place(cozmo.util.degrees(degree)).wait_for_completed()


def getPath(target):
    print("Getting Path")
    path = []
    last = target
    while last.parent is not None:
        path.insert(0, last)
        last = last.parent
    return path


async def driveStep(target, robot: cozmo.robot.Robot):
    print("DriveStep")
    global pose, marked
    dx = target[0] - pose[0]
    dy = target[1] - pose[1]

    degrees = np.arctan2(dy, dx) * (180 / np.pi)
    distance = np.sqrt(dx ** 2 + dy ** 2)

    update_cmap = await find_goal_and_obstacles(robot, marked, Node((pose[0], pose[1])))
    if update_cmap:
        pose = (target[0], target[1], pose[2])
        return False

    await turnToAngle(degrees, robot)

    update_cmap = await find_goal_and_obstacles(robot, marked, Node((pose[0], pose[1])))
    if update_cmap:
        pose = (target[0], target[1], pose[2])
        return False

    await robot.drive_straight(cozmo.util.distance_mm(distance),
                               cozmo.util.speed_mmps(100), in_parallel=True).wait_for_completed()

    update_cmap = await find_goal_and_obstacles(robot, marked, Node((pose[0], pose[1])))
    if update_cmap:
        pose = (target[0], target[1], pose[2])
        return False

    pose = (target[0], target[1], pose[2])
    return True


async def driveToLocation(target, robot: cozmo.robot.Robot):
    print("DriveToLocation")
    path = getPath(target)
    for node in path:
        cmap.set_start(Node((pose[0], pose[1]), None))
        update_cmap = await driveStep(node, robot)
        if not update_cmap:
            return False
    return True


async def gotoCenter(robot: cozmo.robot.Robot):
    print("GoToCenter")
    cmap.clear_goals()
    cmap.add_goal(Node((325, 225)))
    cmap.set_start(Node((pose[0], pose[1])))
    RRT(cmap, cmap.get_start())
    update_cmap = await driveToLocation(cmap._goals[0], robot)
    if not update_cmap:
        cmap.clear_goals()
        cmap.reset()
        await gotoCenter(robot)
    cmap.clear_goals()
    print(robot.pose.position.x + start_x, robot.pose.position.y + start_y)
    return True


def cozmo_current_position(robot):
    global start_x, start_y
    return Node((start_x + robot.pose.position.x, start_y + robot.pose.position.y))


def cozmo_current_degree(robot):
    cur_angle = robot.pose.rotation.angle_z.radians * (180 / np.pi)
    cur_angle = cur_angle % 360
    if cur_angle < 0:
        cur_angle += 360
    return cur_angle


def calculate_global_node(local_angle, local_origin, node):
    cos = np.cos(local_angle)
    sin = np.sin(local_angle)
    global_x = node[0] * cos - node[1] * sin + local_origin[0]
    global_y = node[0] * sin + node[1] * cos + local_origin[1]
    return Node((global_x, global_y))


async def find_goal_and_obstacles(robot, marked, cozmo_pos):
    print("FindGoalandObstacles")
    global cmap
    global final_angle
    global validGoal
    validGoal = False

    # Padding of objects and the robot for C-Space
    cube_padding = 44
    cozmo_padding = 100

    # Flags
    update_cmap = False

    # Time for the robot to detect visible cubes
    sleep(1)
    print("FG1")
    for obj in robot.world.visible_objects:

        if obj.object_id in marked:
            continue

        # Calculate the object pose in G_Arena
        # obj.pose is the object's pose in G_Robot
        # We need the object's pose in G_Arena (object_pos, object_angle)
        dx = obj.pose.position.x - robot.pose.position.x
        dy = obj.pose.position.y - robot.pose.position.y

        object_pos = Node((cozmo_pos.x + dx, cozmo_pos.y + dy))
        object_angle = obj.pose.rotation.angle_z.radians
        print("FG2")
        # The goal cube is defined as robot.world.light_cubes[cozmo.objects.LightCube1Id].object_id
        if robot.world.light_cubes[cozmo.objects.LightCube1Id].object_id == obj.object_id:
            final_angle = object_angle * (180 / np.pi)

            # Calculate the approach position of the object
            local_goal_pos_A = Node((0, cozmo_padding))
            goal_pos_A = calculate_global_node(object_angle, object_pos, local_goal_pos_A)
            local_goal_pos_B = Node((0, -cozmo_padding))
            goal_pos_B = calculate_global_node(object_angle, object_pos, local_goal_pos_B)
            local_goal_pos_C = Node((cozmo_padding, 0))
            goal_pos_C = calculate_global_node(object_angle, object_pos, local_goal_pos_C)
            local_goal_pos_D = Node((-cozmo_padding, 0))
            goal_pos_D = calculate_global_node(object_angle, object_pos, local_goal_pos_D)
            
            goal_pos_list = sorted([get_dist(goal_pos_A, cozmo_current_position(robot)),
                                    get_dist(goal_pos_B, cozmo_current_position(robot)),
                                    get_dist(goal_pos_C, cozmo_current_position(robot)),
                                    get_dist(goal_pos_D, cozmo_current_position(robot))])

            if goal_pos_list[0] == get_dist(goal_pos_A, cozmo_current_position(robot)):
                goal_pos = goal_pos_A
            elif goal_pos_list[0] == get_dist(goal_pos_B, cozmo_current_position(robot)):
                goal_pos = goal_pos_B
            elif goal_pos_list[0] == get_dist(goal_pos_C, cozmo_current_position(robot)):
                goal_pos = goal_pos_C
            elif goal_pos_list[0] == get_dist(goal_pos_D, cozmo_current_position(robot)):
                goal_pos = goal_pos_D
                
            # Check whether this goal location is valid
            print("FG3")
            if cmap.is_inside_obstacles(goal_pos) or (not cmap.is_inbound(goal_pos)):
                print("Invalid goal/cmap")
                continue
            else:
                print("Valid Goal!")
                validGoal = True
                cmap.clear_goals()
                cmap.add_goal(goal_pos)

        # Define an obstacle by its four corners in clockwise order
        obstacle_nodes = [calculate_global_node(object_angle, object_pos, Node((cube_padding, cube_padding))),
                          calculate_global_node(object_angle, object_pos, Node((cube_padding, -cube_padding))),
                          calculate_global_node(object_angle, object_pos, Node((-cube_padding, -cube_padding))),
                          calculate_global_node(object_angle, object_pos, Node((-cube_padding, cube_padding)))]
        cmap.add_obstacle(obstacle_nodes)
        marked[obj.object_id] = obj
        update_cmap = True
            
        if (update_cmap):
            print("ResetCMAP")
            cmap.reset()
            pose = (cozmo_current_position(robot).x, cozmo_current_position(robot).y, cozmo_current_degree(robot))
            find_goal_and_obstacles(robot, marked, pose)
            
    return update_cmap


async def lookForCube(marked, robot: cozmo.robot.Robot):
    update_cmap = await find_goal_and_obstacles(robot, marked, Node((pose[0], pose[1])))
    
    if update_cmap:
        print("new change to the map, clear the path and re-compute")
        cmap.reset()
    await turnDegrees(30, robot)
    return


async def CozmoPlanning(robot: cozmo.robot.Robot):
    # Allows access to map and stopevent, which can be used to see if the GUI
    # has been closed by checking stopevent.is_set()
    global cmap, stopevent
    ########################################################################
    # TODO: please enter your code below.
    # Description of function provided in instructions
    global pose, final_angle, marked
    i = 0

    pose = (cozmo_current_position(robot).x, cozmo_current_position(robot).y, cozmo_current_degree(robot))

    cmap.set_start(Node((pose[0], pose[1])))
    await robot.set_head_angle(cozmo.util.degrees(-10)).wait_for_completed()
    while i < 12:
        await lookForCube(marked, robot)
        i += 1
        if len(cmap._goals) == 1:
            await turnToAngle(0, robot)
            break

    if len(cmap._goals) == 0:
        await gotoCenter(robot)

    while len(cmap._goals) == 0:
        await lookForCube(marked, robot)

    cmap.set_start(Node((pose[0], pose[1])))
    RRT(cmap, cmap.get_start())
    while await driveToLocation(cmap._goals[0], robot) is False:
        cmap.set_start(Node((pose[0], pose[1])))
        RRT(cmap, cmap.get_start())
        sleep(1)

    await turnToAngle(final_angle-90, robot)

    while True:
        sleep(10)


################################################################################
#                     DO NOT MODIFY CODE BELOW                                 #
################################################################################

class RobotThread(threading.Thread):
    """Thread to run cozmo code separate from main thread
    """

    def __init__(self):
        threading.Thread.__init__(self, daemon=True)

    def run(self):
        # Please refrain from enabling use_viewer since it uses tk, which must be in main thread
        cozmo.run_program(CozmoPlanning, use_3d_viewer=False, use_viewer=False)
        stopevent.set()


class RRTThread(threading.Thread):
    """Thread to run RRT separate from main thread
    """

    def __init__(self):
        threading.Thread.__init__(self, daemon=True)

    def run(self):
        while not stopevent.is_set():
            RRT(cmap, cmap.get_start())
            sleep(100)
            cmap.reset()
        stopevent.set()


if __name__ == '__main__':
    global cmap, stopevent
    stopevent = threading.Event()
    cmap = CozMap("maps/emptygrid.json", node_generator)
    robot_thread = RobotThread()
    robot_thread.start()
    visualizer = Visualizer(cmap)
    visualizer.start()
