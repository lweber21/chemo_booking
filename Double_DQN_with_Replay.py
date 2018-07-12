import BookEnv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import ExperienceBuffer
import Qnetwork
import Hyperparameters as param

tf.reset_default_graph()


#instantiante qnetwork and memory
Qnetwork = Qnetwork.Qnetwork(param.STATES, param.N_ACTIONS, hidden=[4], learning_rate=param.ALPHA)
memory = ExperienceBuffer.ExperienceBuffer(param.MEMORY_SIZE)


#create the environment
env = BookEnv.BookingEnv(days=param.DAYS, daily_avail=param.DAILY_CAP, demand_dist=param.DEMAND_DIST)

#
state = env.get_state()
patient = env.get_patient()
for i in range(param.PRE_TRAIN_LENGTH):
    action = env.action_sample()
    next_state, next_patient, reward, done = env.step(action, patient)
    memory.add([state, action, reward, next_state, done])
    state = next_state
    patient = next_patient


def update_explore(decay_step, eps_init=1, eps_final=0.01, decay_rate=0.0001):
    return eps_final + (eps_init - eps_final) * np.exp(-decay_rate * decay_step)


saver = tf.train.Saver()

if param.TRAIN:
    rList = []
    dList = []
    qList = []
    lossValues = []
    decay_step = 1
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for episode in range(param.N_EPISODES):
            env.reset()
            state = env.get_state()

            patient = env.get_patient()

            total_reward = 0
            total_demand = 0
            q = []
            j = 0
            while j <= param.MAX_EPISODE_STEPS:
                j += 1
                a, allQ = sess.run([Qnetwork.predict, Qnetwork.Qout], feed_dict={Qnetwork.input: state})


                if np.random.rand(1) < update_explore(decay_step):
                    a[0] = env.action_sample()
                next_state, next_patient, reward, done = env.step(a[0], patient)
                experience = [state, a[0], reward, next_state, done]
                memory.add(experience)
                decay_step += 1

                #track statistics
                q.append(allQ[0][a[0]])
                total_demand += patient.demand
                total_reward += reward
                patient = next_patient
                state = next_state


                #learn
                batch = memory.sample(param.BATCH_SIZE)
                #print("Batch")
                #print(batch)
                # print(type(batch))
                # print(batch)
                # print(batch.shape)
                batch_states = batch[:, 0]
                batch_actions = batch[:, 1].astype(int)


                batch_rewards = batch[:, 2].reshape(-1,1)
                batch_next_states = batch[:, 3]
                batch_dones = batch[:, 4].reshape(-1, 1)

                batch_states = np.vstack([s for s in batch_states])
                batch_next_states = np.vstack([n for n in batch_next_states])

                Qprime = sess.run(Qnetwork.Qout, feed_dict={Qnetwork.input: batch_next_states})


                maxQprime = np.max(Qprime, axis=1, keepdims=True)
                targetQ = batch_rewards + param.GAMMA * maxQprime


                _, loss, act, q_model, qt = sess.run([Qnetwork.updateModel, Qnetwork.loss, Qnetwork.actions_onehot, Qnetwork.Q, Qnetwork.Qout], feed_dict={Qnetwork.input: batch_states,
                                                                Qnetwork.actions: batch_actions,
                                                                Qnetwork.targetQ: targetQ})
                #
                # print(batch_actions)
                #
                # print("Action one hot")
                # print(act)
                # print("Calcualted Qvalues")
                # print(qt)
                #
                # print("Predicted Q)")
                # print(q_model)
                # print("targetQ")
                # print(targetQ)
                # print("Loss: %f" % loss)

            if episode % 100 == 0:
                save_path = saver.save(sess, "./models/model.ckpt")
                print("Model Saved at Episode: %i" % (episode))
                print(f"total reward -> {total_reward}")
               # print(f"AvqQ -> {np.mean(q)}")

            rList.append(total_reward)
            dList.append(total_demand)
            #print(loss)
            lossValues.append(loss)
            avgQ = np.mean(q)
        save_path = saver.save(sess, "./models/model.ckpt")

    print(np.mean(rList))
    print(np.mean(avgQ))
    print(np.mean(lossValues))
    xvals = np.arange(1, len(rList)+1).tolist()

    plt.plot(xvals, rList)
    plt.show()

    plt.plot(xvals, lossValues)
    plt.show()


with tf.Session() as sess:
    saver.restore(sess, "./models/model.ckpt")
    env.reset()
    state = env.get_state()
    patient = env.get_patient()
    total_rewards = 0
    j = 0
    while j <= param.MAX_EPISODE_STEPS:
        j += 1
        a, allQ = sess.run([Qnetwork.predict, Qnetwork.Qout], feed_dict={Qnetwork.input: state})
        next_state, next_patient, reward, done = env.step(a[0], patient)
        print(state)
        print(a)
        print(allQ)
        total_rewards += reward
        patient = next_patient
        state = next_state

    print(str(env))
    print(f"total reward -> {total_rewards}")
