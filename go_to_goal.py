#!/usr/bin/env python3

import cv2
import cozmo
import numpy as np
from numpy.linalg import inv
import threading
import time

from ar_markers.hamming.detect import detect_markers

from grid import CozGrid
from gui import GUIWindow
from particle import Particle, Robot
from setting import *
from particle_filter import *
from utils import *

# camera params
camK = np.matrix([[295, 0, 160], [0, 295, 120], [0, 0, 1]], dtype='float32')

#marker size in inches
marker_size = 3.5

# tmp cache
last_pose = cozmo.util.Pose(0,0,0,angle_z=cozmo.util.Angle(degrees=0))

# goal location for the robot to drive to, (x, y, theta)
goal = (6,10,0)

# map
Map_filename = "map_arena.json"


async def image_processing(robot):

    global camK, marker_size

    event = await robot.world.wait_for(cozmo.camera.EvtNewRawCameraImage, timeout=30)

    # convert camera image to opencv format
    opencv_image = np.asarray(event.image)
    
    # detect markers
    markers = detect_markers(opencv_image, marker_size, camK)
    
    # show markers
    for marker in markers:
        marker.highlite_marker(opencv_image, draw_frame=True, camK=camK)
        #print("ID =", marker.id);
        #print(marker.contours);
    cv2.imshow("Markers", opencv_image)
    return markers

#calculate marker pose
def cvt_2Dmarker_measurements(ar_markers):
    
    marker2d_list = []
    
    for m in ar_markers:
        R_1_2, J = cv2.Rodrigues(m.rvec)
        R_1_1p = np.matrix([[0,0,1], [0,-1,0], [1,0,0]])
        R_2_2p = np.matrix([[0,-1,0], [0,0,-1], [1,0,0]])
        R_2p_1p = np.matmul(np.matmul(inv(R_2_2p), inv(R_1_2)), R_1_1p)
        #print('\n', R_2p_1p)
        yaw = -math.atan2(R_2p_1p[2,0], R_2p_1p[0,0])
        
        x, y = m.tvec[2][0] + 0.5, -m.tvec[0][0]
        # print('x =', x, 'y =', y,'theta =', yaw)
        
        # remove any duplate markers
        dup_thresh = 2.0
        find_dup = False
        for m2d in marker2d_list:
            if grid_distance(m2d[0], m2d[1], x, y) < dup_thresh:
                find_dup = True
                break
        if not find_dup:
            marker2d_list.append((x,y,math.degrees(yaw)))

    return marker2d_list


#compute robot odometry based on past and current pose
def compute_odometry(curr_pose, cvt_inch=True):
    global last_pose
    last_x, last_y, last_h = last_pose.position.x, last_pose.position.y, \
        last_pose.rotation.angle_z.degrees
    curr_x, curr_y, curr_h = curr_pose.position.x, curr_pose.position.y, \
        curr_pose.rotation.angle_z.degrees

    if cvt_inch:
        last_x, last_y = last_x / 25.6, last_y / 25.6
        curr_x, curr_y = curr_x / 25.6, curr_y / 25.6

    return [[last_x, last_y, last_h],[curr_x, curr_y, curr_h]]

#particle filter functionality
class ParticleFilter:

    def __init__(self, grid):
        self.particles = Particle.create_random(PARTICLE_COUNT, grid)
        self.grid = grid

    def update(self, odom, r_marker_list):

        # ---------- Motion model update ----------
        self.particles = motion_update(self.particles, odom)

        # ---------- Sensor (markers) model update ----------
        self.particles = measurement_update(self.particles, r_marker_list, self.grid)

        # ---------- Show current state ----------
        # Try to find current best estimate for display
        m_x, m_y, m_h, m_confident = compute_mean_pose(self.particles)
        return (m_x, m_y, m_h, m_confident)

def checkConverge(particles):
    #calculate the mean pose
    m_x, m_y, m_h, m_c = compute_mean_pose(particles)
    converge_count = 0
    print("enter checking convergence function")
    #for all the particles
    for p in particles:
        gridDistance = grid_distance(p.x, p.y, m_x, m_y)
        headingDegree = diff_heading_deg(p.h, m_h)
        
        if gridDistance <= 0.6 and headingDegree < 10:
            converge_count = converge_count + 1
    
    #calculate the converge
    particleConverge = converge_count >= 0.97 * PARTICLE_COUNT

    return particleConverge

async def run(robot: cozmo.robot.Robot):
    global last_pose
    global grid, gui

    # start streaming
    robot.camera.image_stream_enabled = True

    #start particle filter
    pf = ParticleFilter(grid)

    # ############################################################################
    # ######################### YOUR CODE HERE####################################
    
    #has the goal been marked?
    goalMarked = False
    
    #whats the current state
    curState = 0
    
    while True:
        if not checkConverge(pf.particles):
            while(robot.is_picked_up):
                #this loop is for the kidnapped robot problem (robot picked up)
                #robot acts angry
                await robot.play_anim_trigger(cozmo.anim.Triggers.DizzyReactionHard).wait_for_completed()
                print("Robot is dizzy from being picked up")
                
                #reset pose
                last_pose = robot.pose
                
                #set curState to 0
                curState = 0
                
                #set particle filter
                pf = ParticleFilter(grid)
            else:
                #robot is put back down
                #robot is happy to be put down
                #await robot.play_anim_trigger(cozmo.anim.Triggers.CodeLabReactHappy).wait_for_completed()
                
                #first reset the robots pose
                last_pose = robot.pose
                
                #now find odometry using newly updated pose
                robotOdometry = compute_odometry(last_pose)
                
                #find the markers now
                markerGoals = await image_processing(robot)
                
                #calculate the measurements (obtain list of currently seen markers and their poses)
                markerMeasurement = cvt_2Dmarker_measurements(markerGoals)
                
                #with markerMeasurement and robotOdomoetry, update the particle filter
                pfUpdate = pf.update(robotOdometry, markerMeasurement)
                
                #update pf GUI for debugging
                gui.show_particles(pf.particles)
                gui.show_mean(pfUpdate[0], pfUpdate[1], pfUpdate[2], pfUpdate[3])
                gui.updated.set()
                
                if robot.is_picked_up:
                    pf = ParticleFilter(grid)
                    continue
                
                #drive to markers
                #while there are markers in the marker list, continue to drive
                #if(len(markerGoals) != 0 and markerMeasurement[0][0] > 2.0):
                if(pfUpdate[3] == True):
                    curState = 1
                    print("robot drives straight, found marker")
                    #drive to marker, animate to show marker has been found, continue
                    await robot.drive_straight(cozmo.util.distance_mm(40), cozmo.util.speed_mmps(40)).wait_for_completed()
                else:
                    print("robot is searching")
                    await robot.turn_in_place(cozmo.util.degrees(20)).wait_for_completed()
        else:
            #while the robot has not reached the goal
            if not goal_reached:
                print("goal still has not been reached")
                
                #find the new mean pose
                m_x, m_y, m_h, m_c = compute_mean_pose(pf.particles)
                
                #convert goal to inches
                goalInches = (goal[0]*25/25.6, goal[1]*25/25.6, goal[2])
                dif_y = goalInches[1] - m_y
                dif_x = goalInches[0] - m_x
                
                #find the angle
                angleCalc = math.degrees(math.atan2(dif_y, dif_x))
                dist = math.sqrt(dif_y**2 + dif_x**2) * 25.6
                angle = diff_heading_deg(angleCalc, m_h)

                #track the drive distance
                driveDist = 0

                #check to see if the robot is delocalized while driving
                delocalized = False
                while driveDist < dist:
                    if robot.is_picked_up:
                        #robot has been picked up, reset particle filter
                        pf = ParticleFilter(grid)
                        delocalized = True
                        break
                    #otherwise drive straight
                    await robot.drive_straight(cozmo.util.distance_mm(min(40, dist-d)), cozmo.util.speed_mmps(40)).wait_for_completed()
                    driveDist = driveDist + 40
                
                if delocalized:
                    continue

                #if the robot is picked up, reset the particle filter
                if robot.is_picked_up:
                    pf = ParticleFilter(grid)
                    continue
                else:
                    await robot.play_anim_trigger(cozmo.anim.Triggers.AcknowledgeObject).wait_for_completed()
                    goal_reached = True
            else:
                print("robot reached the goal")
                time.sleep(1)
                #robot drives straight
                await robot.drive_straight(cozmo.util.distance_mm(0), cozmo.util.speed_mmps(40)).wait_for_completed()
                
                #if the robot is picked up, reset particle filter
                if robot.is_picked_up:
                    print ("robot is picked up")
                    goal_reached = False
                    pf = ParticleFilter(grid)
                    continue

    ############################################################################


class CozmoThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self, daemon=False)

    def run(self):
        cozmo.run_program(run, use_viewer=False)


if __name__ == '__main__':

    # cozmo thread
    cozmo_thread = CozmoThread()
    cozmo_thread.start()

    # init
    grid = CozGrid(Map_filename)
    gui = GUIWindow(grid)
    #gui.start()
