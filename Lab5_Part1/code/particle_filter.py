from grid import *
from particle import Particle
from utils import *
from setting import *

# ------------------------------------------------------------------------
def motion_update(particles, odom):
    """ Particle filter motion update
        Arguments:
        particles -- input list of particle represents belief p(x_{t-1} | u_{t-1})
                before motion update
        odom -- odometry to move (dx, dy, dh) in *robot local frame*
        Returns: the list of particles represents belief \tilde{p}(x_{t} | u_{t})
                after motion update
    """
    motion_particles = []
    if odom[0] == 0 and odom[1] == 0 and odom[3] == 0:
        return particles
    for particle in particles:
        dx = odom[1][0] - odom[0][0]
        dx = add_gaussian_noise(dx, ODOM_TRANS_SIGMA)
        dy = odom[1][1] - odom[0][1]
        dy = add_gaussian_noise(dy, ODOM_TRANS_SIGMA)
        dh = odom[1][2] - odom[0][2]
        dh = add_gaussian_noise(dh, ODOM_HEAD_SIGMA)
        x = particle.x + dx
        y = particle.y + dy
        h = particle.h + dh
        motion_particles.append(Particle(x, y, h))
    return motion_particles

# ------------------------------------------------------------------------
def measurement_update(particles, measured_marker_list, grid):
    """ Particle filter measurement update

        Arguments: 
        particles -- a list of particle represents belief \tilde{p}(x_{t} | u_{t})
                before measurement update
        measured_marker_list -- robot detected marker list, each marker has format:
                measured_marker_list[i] = (rx, ry, rh)
                rx -- marker's relative X coordinate in robot's frame
                ry -- marker's relative Y coordinate in robot's frame
                rh -- marker's relative heading in robot's frame, in degree
        grid -- grid world map containing the marker information. 
                see grid.py and CozGrid for definition

        Returns: the list of particle representing belief p(x_{t} | u_{t})
                after measurement update
    """
    weight_particles = []
    
    #increment through all the particles
    for parts in particles:
        #check the grid
        if grid.is_free(*p.xy):
            if len(measured_marker_list) > 0:
                #set particle markers and weight
                particle_markers = p.read_markers(grid)
                weight = marker_multi_measurement_model(measured_marker_list, p_marker_list)
                #set weight accordingly
            else:
                weight = 1
        else:
            weight = 0
        #update the weight particles list
        weight_particles.append(weight)
    
    measured_particles = [] 

    #set the normalized weight for particle weights
    normalized_weight = sum(weight for weight in particle_weights)
    #calculate the avg
    average_weight = normalized_weight / len(particles)

    #iterate throught the particle weights and count the number of zeros
    i = 0
    for weight in particle_weights:
        if weight == 0:
            i = i + 1

    #calc avg weight for particles
    if i != len(particles):
        average_weight = average_weight / (len(particles)-i) * len(particles)

    weights_normalized = []
    #normalize the weights for the particles
    if normalized_weight:
        for weight in particle_weights:
            weights_normalized.append(weight / normalized_weight)

    #calc for weighted distribution
    distribution = WeightedDistribution(particles, weights_normalized)

    #calc for measure particles
    for _ in particles:
        parts = distribution.pick()
        if parts is None or avg_weight < min_avg_weight:
            new_particle = Particle.create_random(1, grid)[0]
        else:
            new_particle = Particle(p.x, p.y, p.h)
        measured_particles.append(new_particle)
    
    return measured_particles
