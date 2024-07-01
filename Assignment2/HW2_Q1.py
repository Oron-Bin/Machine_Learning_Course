'''
oron binyamin 208306274 gilad dolev 206325334
The code organizes the limitations of the force, distance and velocity of the given problem.
It sets each one of them into 3 categories.
Then it sets rules in order to provide the output of the force, in order to achieve the goal.
Since the idea is to make the car maintain distance between the range of 7.5m to 8.5m
and the initial distance (y0) an be between 5m to 20m, there're orders to "go full force", and "max slow",
in order to get between the range before the last 50m of the car.
There's an error function that comes up, in case the car doesn't stand in the terms.
'''

import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from two_cars import two_cars

velocity_input = ctrl.Antecedent(np.arange(-1, 6, 0.1), 'Velocity')
distance_input = ctrl.Antecedent(np.arange(-1, 20, 0.1), 'Distance')
force_output = ctrl.Consequent(np.arange(-1500, 3001, 1), 'Force')

distance_input['Close'] = fuzz.trimf(distance_input.universe, [0, 8, 8])
distance_input['Ideal'] = fuzz.trimf(distance_input.universe, [7.5, 8, 8.5])
distance_input['Far'] = fuzz.trimf(distance_input.universe, [8, 8.49, 20])

velocity_input['Slow'] = fuzz.trimf(velocity_input.universe, [0, 0, 3])
velocity_input['Medium'] = fuzz.trimf(velocity_input.universe, [2.3, 3, 3.7])
velocity_input['Fast'] = fuzz.trimf(velocity_input.universe, [3, 6, 6])

force_output['Low'] = fuzz.trimf(force_output.universe, [-1500, -1500, 0])
force_output['Medium'] = fuzz.trimf(force_output.universe, [-750, 0, 750])
force_output['High'] = fuzz.trimf(force_output.universe, [0, 3000, 3000])

rule1 = ctrl.Rule(distance_input['Close'] & velocity_input['Fast'], force_output['Low'])
rule2 = ctrl.Rule(distance_input['Close'] & velocity_input['Medium'], force_output['Low'])
rule3 = ctrl.Rule(distance_input['Close'] & velocity_input['Slow'], force_output['Medium'])
rule4 = ctrl.Rule(distance_input['Ideal'] & velocity_input['Fast'], force_output['Low'])
rule5 = ctrl.Rule(distance_input['Ideal'] & velocity_input['Medium'], force_output['Medium'])
rule6 = ctrl.Rule(distance_input['Ideal'] & velocity_input['Slow'], force_output['Medium'])
rule7 = ctrl.Rule(distance_input['Far'] & velocity_input['Fast'], force_output['Medium'])
rule8 = ctrl.Rule(distance_input['Far'] & velocity_input['Medium'], force_output['High'])
rule9 = ctrl.Rule(distance_input['Far'] & velocity_input['Slow'], force_output['High'])

force_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
force_sim = ctrl.ControlSystemSimulation(force_ctrl)

def fuzzy_controller(velocity, distance, sim):
    try:
        sim.input['Velocity'] = velocity
        sim.input['Distance'] = distance
        sim.compute()
        force = sim.output['Force']
        return force
    except ValueError:
        print(f"Error: Input values (velocity={velocity}, distance={distance}) are outside the valid range.")

def running_func(cars):
    i = 0
    time = []
    position_car1 = []
    position_car2 = []
    distance = []
    velocity = []
    force = []
    arriving_50_m_time = 0
    while cars.x < cars.road_Length and cars.t < 60:

        time.append(cars.t)
        position_car1.append(cars.x)
        position_car2.append(cars.x_lead)
        distance.append(cars.distance)
        velocity.append(cars.v)

        f = fuzzy_controller(cars.v, cars.distance, force_sim)

        # Check if the condition is not satisfied for the last 50m
        if cars.x >= cars.road_Length - 50 and abs(cars.distance - 8) >= 0.5:
            raise ValueError('The distances between the cars is not between the range.')

        # Adjust the force to optimize the system
        if cars.distance >= 17:
            f = 3000  # Limits the force to a max of 3000N
        if cars.v >= 5 and cars.distance < 17:
            f = -1500 # Limits the force to a min of -1500N
        if cars.v >= 3 and cars.distance <= 7.5:
            f = -1500 # Limits the force to a min of -1500N
        if round(cars.x, 1) == 50.0:
            arriving_50_m_time = cars.t
        success = cars.step(f)

        if not success:
            break

        if i % 50 == 0:
            cars.draw()

        i += 1
        force.append(f)
    fig, axs = plt.subplots(4, figsize=(12, 20))
    fig.subplots_adjust(hspace=1.5)

    # Location of the cars.
    axs[0].plot(time, position_car1, label='Car 1')
    axs[0].plot(time, position_car2, label='Car 2')
    axs[0].axvline(x=arriving_50_m_time, color='r', linestyle='--')
    axs[0].set_xlabel('Time [sec]')
    axs[0].set_ylabel('Location [m]')
    axs[0].set_title('Locations of the cars')
    axs[0].legend()

    # The distance between the cars.
    axs[1].plot(time, distance)
    axs[1].axvline(x=arriving_50_m_time, color='r', linestyle='--')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Distance')
    axs[1].set_title('Distance between the cars')

    # Velocity of car 1.
    axs[2].plot(time, velocity)
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Velocity')
    axs[2].set_title('Velocity of Car 1')

    # Force of car 1.
    axs[3].plot(time, force)
    axs[3].axvline(x=arriving_50_m_time, color='r', linestyle='--')
    axs[3].set_xlabel('Time [sec]')
    axs[3].set_ylabel('Force [N]')
    axs[3].set_title('Force on Car 1')

    plt.show()

## Change y0 here, in order to check the code.
if __name__ == '__main__':
    y0_input = 10
    Cars = two_cars(y0=y0_input)
    running_func(Cars)

# Create a figure with three subplots.
fig, axs = plt.subplots(1, 3, figsize=(12, 4))

# Plot velocity membership functions.
axs[0].plot(velocity_input.universe, fuzz.trimf(velocity_input.universe, [0, 0, 3]), label='Slow')
axs[0].plot(velocity_input.universe, fuzz.trimf(velocity_input.universe, [2.3, 3, 3.7]), label='Medium')
axs[0].plot(velocity_input.universe, fuzz.trimf(velocity_input.universe, [3, 6, 6]), label='Fast')
axs[0].set_xlabel('Velocity [m/sec]')
axs[0].set_title('Membership functions of Velocity')
axs[0].legend()

# Plot distance membership functions
axs[1].plot(distance_input.universe, fuzz.trimf(distance_input.universe, [0, 8, 8]), label='Close')
axs[1].plot(distance_input.universe, fuzz.trimf(distance_input.universe, [7.5, 8, 8.5]), label='Ideal')
axs[1].plot(distance_input.universe, fuzz.trimf(distance_input.universe, [8, 8.49, 20]), label='Far')
axs[1].set_xlabel('Distance [m]')
axs[1].set_title('Membership functions of Distance')
axs[1].legend()

# Plot force membership functions
axs[2].plot(force_output.universe, fuzz.trimf(force_output.universe, [-1500, -1500, 0]), label='Low')
axs[2].plot(force_output.universe, fuzz.trimf(force_output.universe, [-750, 0, 750]), label='Medium')
axs[2].plot(force_output.universe, fuzz.trimf(force_output.universe, [0, 3000, 3000]), label='High')
axs[2].set_xlabel('Force [N]')
axs[2].set_title('Membership functions of Force')
axs[2].legend()

plt.tight_layout()
plt.show()
