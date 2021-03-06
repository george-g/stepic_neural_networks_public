# from HW_3.cars import *
from cars.world import SimpleCarWorld
from cars.agent import SimpleCarAgent
from cars.physics import SimplePhysics
from cars.track import generate_map
import numpy as np
import random

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--steps", type=int)
parser.add_argument("-f", "--filename", type=str) #, default = 'network_config_agent_0_layers_25_25_60_60_60_1.txt')
parser.add_argument("-e", "--evaluate", type=bool)
parser.add_argument("--seed", type=int)
parser.add_argument('--hiddenlayers', dest='hiddenLayers', metavar='N', type=int, nargs='+',
                    help='amount of neurons in hidden layers')
parser.add_argument('--rays', dest='rays', type=int, 
                    help='amount of ladar ray', default = 21)
args = parser.parse_args()

print(args.steps, args.seed, args.filename, args.evaluate)

steps = args.steps
seed = args.seed if args.seed else 23
np.random.seed(seed)
random.seed(seed)
m = generate_map(8, 5, 3, 3)

hiddenLayersList = args.hiddenLayers if args.hiddenLayers else [] # [55, 45]

if args.filename:
    agent = SimpleCarAgent.from_file(args.filename)
    w = SimpleCarWorld(1, m, SimplePhysics, SimpleCarAgent, timedelta=0.2)
    if args.evaluate:
        print(w.evaluate_agent(agent, steps))
    else:
        w.set_agents([agent])
        w.run(steps)
else:
    SimpleCarWorld(1, m, SimplePhysics, SimpleCarAgent, timedelta=0.2, rays=args.rays, hiddenLayers=hiddenLayersList).run(steps)
