# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import cv2
import hydra
import pybullet as p
import pybulletX as px
import tacto
import numpy as np
import json
import itertools

from tf.transformations import quaternion_from_matrix, quaternion_matrix, euler_from_quaternion, quaternion_from_euler

log = logging.getLogger(__name__)

# Load the config YAML file from examples/conf/digit.yaml
@hydra.main(config_path="conf", config_name="digit")
def main(cfg):

    # Initialize digits
    bg = cv2.imread("conf/resized.jpg")
    digits = tacto.Sensor(**cfg.tacto, background=bg)
   
    # Initialize World
    log.info("Initializing world")
    px.init(mode=p.GUI)
    p.resetDebugVisualizerCamera(**cfg.pybullet_camera)

    # Create and initialize DIGIT
    digit_body = px.Body(**cfg.digit)
    digits.add_camera(digit_body.id, [-1])

    body_index = 'pen' # Run only this body
    bodies = {'knight':cfg.knight, 'pawn':cfg.pawn, 'queen':cfg.queen, 'rook':cfg.rook, 'bishop':cfg.bishop, 'king':cfg.king, 'pen':cfg.pen}
    bodies = {body_index: bodies[body_index]}
    p.setGravity(0,0,-1)

    count = 0 # Total number of images
    for body_key, body in bodies.items(): 
        json_dict = []
        body = px.Body(**body)
        digits.add_body(body)
        home = [-150,0,60] # Center of the sensor
        for qwiggle in range(-4,5,5): # How flat the object sits
            for x in range(home[0]-80,home[0]+90,40): # Offset x
                for y in range(home[1]-40,home[1]+50,40): # Offset y
                    for qy in range(-100,100,10): # Rotation about z while laying down
                        # Two views to extract views on both sides of the object
                        for qz in itertools.chain(range(-20, 30, 10), range(80, 130, 10)): # Around z (standing up)
                            chances = -1 # Wait for contact 
                            for z in range(home[2]+50,home[2]-30,-2): # Offset z
                                if chances == 0: # Once contact is made, 6 chances to get another valid image
                                    break
                                p.resetBasePositionAndOrientation(
                                    body.id, [x/10000, y/10000, z/2000], 
                                    quaternion_from_euler(np.pi/2+qwiggle/80,qy/100*np.pi, -qz/100*np.pi, 'rxyz'))
                                p.stepSimulation()
                                color, depth = digits.render()
                                if np.quantile(depth, 0.95) < .00015 or np.quantile(depth, 0.95) > 0.0022:
                                    chances -= 1
                                    continue 
                                chances = 6
                                colors = np.concatenate(color, axis=1)
                                json_dict.append({"x":x/10000, 
                                            "y":y/10000, 
                                            "z":z/1000, 
                                            "i":np.pi/2+qwiggle/80, 
                                            "j":qy/100*np.pi, 
                                            "k":qz/100*np.pi,
                                            "RGB_image": f"RGB_{body_key}{count}.jpg",
                                            "Depth_image": f"Depth_{body_key}{count}.tiff",
                                            "Max_depth": float(np.quantile(depth, 0.95)),
                                            })
                                digits.updateGUI(color, depth)

                                depth = np.asarray(depth) /0.01 * np.iinfo(np.uint16).max

                                depth = depth.astype(np.uint16)[0]
                                cv2.imwrite("../data/" + json_dict[-1]["Depth_image"], depth)
                                cv2.imwrite("../data/" + json_dict[-1]["RGB_image"], colors)

                               
                                count += 1
                                print(body_index + ': ' + str(count), end='\r')

        p.removeBody(body.id) # Remove body from simulation
        with open(f'../data//{body_key}.json', 'w') as f:
            json.dump(json_dict, f) # Save json files

if __name__ == "__main__":
    main()
