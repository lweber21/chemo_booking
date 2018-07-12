import numpy as np
import csv


class Patient:

    def __init__(self, patient_id, demand):
        self.patient_id = patient_id
        self.demand = demand


class BookingEnv:

    def __init__(self, days, daily_avail=5, cap_init=0, demand_dist=[1, 1]):
        self.days = days
        self.demandDist = demand_dist
        self.total_demand = 0
        self.patient_id = 0
        self.daily_avail = daily_avail
        self.cap_init = cap_init
        self.cap_avail = np.empty(self.days)
        self.cap_avail.fill(self.daily_avail)

        self.cap_used = np.empty(self.days)
        self.cap_used.fill(self.cap_init)
        self.booking_list = [[] for _ in range(days)]
        self.overtime = 0
        self.set_overtime()
        self.Patient = None
        self.generate_patient()
        self.state = None
        self.set_state()

    def reset(self):
        #resets the environment back to original initialization settings
        self.total_demand = 0
        self.cap_avail = np.empty(self.days)
        self.cap_avail.fill(self.daily_avail)
        self.cap_used = np.empty(self.days)
        self.cap_used.fill(self.cap_init)
        self.booking_list = [[] for _ in range(self.days)]
        self.set_overtime()
        self.generate_patient()
        self.set_state()

    def action_sample(self):
        return np.random.choice(range(self.days))

    def step(self, action, patient):
        done = False
        #done_state = np.array([0] * self.days)
        self.book_patient(action, patient)
        reward = self.reward(action)
        self.set_overtime()
        self.generate_patient()
        self.set_state()
        if False:
            done = True

        return self.get_state(), self.get_patient(), reward, done

    def set_total_demand(self, demand):
        self.total_demand += demand

    def book_patient(self, action, patient):
        #books a patient with demand d to a specified day
        #print(action)
        #print(self.booking_list)
        #self.booking_list[action].append(patient_id)
        #print(self.booking_list)
        self.cap_used[action] += patient.demand
        self.set_total_demand(patient.demand)

    def reward(self, action):
        if self.overtime[action] > 0:
            reward = -3 * self.overtime[action]
        elif self.cap_used[action]/self.cap_avail[action] <= 0.5:
            reward = 2.
                     #/(self.cap_used[action]/self.cap_avail[action] + .001)
        else:
            reward = 0
                     #/(self.cap_used[action]/self.cap_avail[action] + .0001)
        return reward

    def generate_patient(self):
        self.patient_id += 1
        self.Patient = Patient(self.patient_id, np.random.choice(self.demandDist))

    def set_state(self):
        current_state = self.cap_avail - self.cap_used
        #zero_array = np.zeros(self.cap_used.shape)
        #current_state = np.maximum(self.cap_avail - self.cap_used , zero_array)
        self.state = np.append(current_state, np.array(self.Patient.demand))
        self.state = self.state.reshape(1, -1)

    def get_state(self):
        return self.state

    def get_patient(self):
        return self.Patient

    def set_overtime(self):
        zero_array = np.zeros(self.cap_used.shape)
        self.overtime = np.maximum(self.cap_used - self.cap_avail, zero_array)

    def write_env(self, fname):
        with open(fname, 'w', newline='') as f:
            thewriter = csv.writer(f)

            thewriter.writerow(["day", "cap_avail", "cap_used", "overtime"])
            for d in range(self.days):
                thewriter.writerow([d+1, self.cap_avail[d], self.cap_used[d], self.overtime[d]])

    def __str__(self):
        result = ""
        for day in range(len(self.cap_avail)):
            result += "Day:" + str(day+1) + " "
            result += "cap_avail:" + str(self.cap_avail[day]) + " "
            result += "cap_used:" + str(self.cap_used[day]) + " "
            result += "overtime:" + str(self.overtime[day]) + "\n"
        return result
        '''if not self.booking_list:
             print("booking_list:" + str(self.booking_list[day]))'''


class Drug:
    def __init__(self, drug_name, hd=False, gi=False, pd=False, exp=False):
        self.drug_name = drug_name
        self.hd = hd
        self.gi = gi
        self.pd = pd
        self.exp = exp

    def __str__(self):
        att_string = " "
        if self.hd:
            att_string += "HD"
        if self.gi:
            att_string += "GI"
        if self.pd:
            att_string += "PD"
        if self.exp:
            att_string += "EXP"

        return "Drug: %s" + att_string % self.drug_name


class Protocol:
    def __init__(self, protocol_name, pattern=[], drugs=[], demand={}):
        self.protocol_name = protocol_name
        self.pattern = pattern
        self.drugs = drugs
        self.demand = demand


''''
testEnv = BookingEnv(days=10, daily_avail=5, cap_init=0, demand_dist=[1,2,3])
print(testEnv)
for i in range(5):
    testEnv.step(np.random.choice(range(testEnv.days)), testEnv.Patient)
    print(testEnv)
'''