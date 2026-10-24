from env.base import MetaEnv
from env.mec_offloaing_envs.offloading_task_graph import OffloadingTaskGraph

from samplers.vectorized_env_executor import MetaIterativeEnvExecutor
import numpy as np
import os

"""
import math for exp() function
"""
import math


class Resources(object):
    """
    This class denotes the MEC server and Mobile devices (computation resources)

    Args:
        mec_process_capable: computation capacity of the MEC server
        mobile_process_capable: computation capacity of the mobile device
        bandwidth_up: wireless uplink band width
        bandwidth_dl: wireless downlink band width
    """

    def __init__(self, mec_process_capable,
                  mobile_process_capable, bandwidth_up = 7.0, bandwidth_dl = 7.0):
        self.mec_process_capble = mec_process_capable
        self.mobile_process_capable = mobile_process_capable
        self.mobile_process_avaliable_time = 0.0
        self.mec_process_avaliable_time = 0.0

        self.bandwidth_up = bandwidth_up
        self.bandwidth_dl = bandwidth_dl

    def up_transmission_cost(self, data):
        rate = self.bandwidth_up * (1024.0 * 1024.0 / 8.0)

        transmission_time = data / rate

        return transmission_time

    def reset(self):
        self.mec_process_avaliable_time = 0.0
        self.mobile_process_avaliable_time = 0.0

    def dl_transmission_cost(self, data):
        rate = self.bandwidth_dl * (1024.0 * 1024.0 / 8.0)
        transmission_time = data / rate

        return transmission_time

    def locally_execution_cost(self, data):
        return self._computation_cost(data, self.mobile_process_capable)

    def mec_execution_cost(self, data):
        return self._computation_cost(data, self.mec_process_capble)

    def _computation_cost(self, data, processing_power):
        computation_time = data / processing_power

        return computation_time

class OffloadingEnvironment(MetaEnv):
    def __init__(self, resource_cluster, batch_size,
                 graph_number,
                 graph_file_paths, time_major):
        self.resource_cluster = resource_cluster

        """
        Failure_rate field added for Reliability computation (only for uplink, server, downlink)
        """
        self.failure_rate_mec = 0.00006
        self.failure_rate_local = 0.0

        self.task_graphs_batchs = []
        self.encoder_batchs = []
        self.encoder_lengths = []
        self.decoder_full_lengths = []
        self.max_running_time_batchs = []
        self.min_running_time_batchs = []

        """
        max/min_reliability_batchs
        """
        self.max_reliability_batchs = []
        self.min_reliability_batchs = []

        self.graph_file_paths = graph_file_paths

        # load all the task graphs into the evnironment
        for graph_file_path in graph_file_paths:
            encoder_batchs, encoder_lengths, task_graph_batchs, decoder_full_lengths, max_running_time_batchs, min_running_time_batchs, \
                max_reliability_batchs, min_reliability_batchs = \
                self.generate_point_batch_for_random_graphs(batch_size, graph_number, graph_file_path, time_major)

            self.encoder_batchs += encoder_batchs
            self.encoder_lengths += encoder_lengths
            self.task_graphs_batchs += task_graph_batchs
            self.decoder_full_lengths += decoder_full_lengths
            self.max_running_time_batchs += max_running_time_batchs
            self.min_running_time_batchs += min_running_time_batchs

            self.max_reliability_batchs += max_reliability_batchs
            self.min_reliability_batchs += min_reliability_batchs

        self.total_task = len(self.encoder_batchs)
        self.optimal_solution = -1
        self.task_id = -1
        self.time_major = time_major
        self.input_dim = np.array(encoder_batchs[0]).shape[-1]

        # set the file paht of task graphs.
        self.graph_file_paths = graph_file_paths
        self.graph_number = graph_number


        self.local_exe_time, _ = self.get_all_locally_execute_time()
        self.mec_exe_time, _ = self.get_all_mec_execute_time()

    def sample_tasks(self, n_tasks):
        """
        Samples task of the meta-environment

        Args:
            n_tasks (int) : number of different meta-tasks needed

        Returns:
            tasks (list) : an (n_tasks) length list of tasks
        """
        return np.random.choice(np.arange(self.total_task), n_tasks, replace=False)

    def merge_graphs(self):
        encoder_batchs = []
        encoder_lengths = []
        task_graphs_batchs = []
        decoder_full_lengths =[]
        max_running_time_batchs = []
        min_running_time_batchs = []

        max_reliability_batchs = []
        min_reliability_batchs = []

        for encoder_batch, encoder_length, task_graphs_batch, \
            decoder_full_length, max_running_time_batch, \
            min_running_time_batch, max_reliability_batch, \
            min_reliability_batch in zip(self.encoder_batchs, self.encoder_lengths,
                                          self.task_graphs_batchs, self.decoder_full_lengths,
                                          self.max_running_time_batchs, self.min_running_time_batchs,
                                          self.max_reliability_batchs, self.min_reliability_batchs):
            encoder_batchs += encoder_batch.tolist()
            encoder_lengths += encoder_length.tolist()
            task_graphs_batchs += task_graphs_batch
            decoder_full_lengths += decoder_full_length.tolist()
            max_running_time_batchs += max_running_time_batch
            min_running_time_batchs += min_running_time_batch

            max_reliability_batchs += max_reliability_batch
            min_reliability_batchs += min_reliability_batch

        self.encoder_batchs = np.array([encoder_batchs])
        self.encoder_lengths = np.array([encoder_lengths])
        self.task_graphs_batchs = [task_graphs_batchs]
        self.decoder_full_lengths = np.array([decoder_full_lengths])
        self.max_running_time_batchs = np.array([max_running_time_batchs])
        self.min_running_time_batchs = np.array([min_running_time_batchs])

        self.max_reliability_batchs = np.array([max_reliability_batchs])
        self.min_reliability_batchs = np.array([min_reliability_batchs])

    def set_task(self, task):
        """
        Sets the specified task to the current environment

        Args:
            task: task of the meta-learning environment
        """
        self.task_id = task

    def get_task(self):
        """
        Gets the task that the agent is performing in the current environment

        Returns:
            task: task of the meta-learning environment
        """
        return self.graph_file_paths[self.task_id]

    """
    change in step() function: reliability added
    """

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the agent

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        plan_batch = []
        task_graph_batch = self.task_graphs_batchs[self.task_id]
        max_running_time_batch = self.max_running_time_batchs[self.task_id]
        min_running_time_batch = self.min_running_time_batchs[self.task_id]

        """Reliability Computation:
            Normalization:
                min_reliability
                max_reliability
                computation in calculate_max_min_runningcost function invoked in generate_point_batch_for_random_graphs
        """
        max_reliability_batch = self.max_reliability_batchs[self.task_id]
        min_reliability_batch = self.min_reliability_batchs[self.task_id]

        for action_sequence, task_graph in zip(action, task_graph_batch):
            plan_sequence = []

            for action, task_id in zip(action_sequence, task_graph.prioritize_sequence):
                plan_sequence.append((task_id, action))

            plan_batch.append(plan_sequence)

        reward_batch, task_finish_time, total_reliability_batch = self.get_reward_batch_step_by_step(plan_batch,
                                                  task_graph_batch,
                                                  max_running_time_batch,
                                                  min_running_time_batch,
                                                  max_reliability_batch,
                                                  min_reliability_batch)

        done = True
        observation = np.array(self.encoder_batchs[self.task_id])
        info = task_finish_time

        rel_info = total_reliability_batch

        return observation, reward_batch, done, info, rel_info

    def reset(self):
        """Resets the state of the environment and returns an initial observation.

        Returns:
            observation (object): the initial observation.
        """
        # reset the resource environment.
        self.resource_cluster.reset()

        return np.array(self.encoder_batchs[self.task_id])

    def render(self, mode='human'):
        pass

    """
    change in generate_point_batch_for_random_graphs
    """

    def generate_point_batch_for_random_graphs(self, batch_size, graph_number, graph_file_path, time_major):
        encoder_list = []
        task_graph_list = []

        encoder_batchs = []
        encoder_lengths = []
        task_graph_batchs = []
        decoder_full_lengths = []

        max_running_time_vector = []
        min_running_time_vector = []

        """
        storing max, min reliability
        """

        max_reliability_vector = []
        min_reliability_vector = []

        max_running_time_batchs = []
        min_running_time_batchs = []

        """
        storing max, min reliability batches (for graphs in a batch)
        """
        max_reliability_batchs = []
        min_reliability_batchs = []

        for i in range(graph_number):
            task_graph = OffloadingTaskGraph(graph_file_path + str(i) + '.gv')
            task_graph_list.append(task_graph)

            max_time, min_time, max_reliability, min_reliability = self.calculate_max_min_runningcost(task_graph.max_data_size,
                                                                    task_graph.min_data_size)
            max_running_time_vector.append(max_time)
            min_running_time_vector.append(min_time)

            """
            storing max, min reliability
            """
            max_reliability_vector.append(max_reliability)
            min_reliability_vector.append(min_reliability)

            # the scheduling sequence will also store in self.'prioritize_sequence'
            scheduling_sequence = task_graph.prioritize_tasks(self.resource_cluster)

            task_encode = np.array(task_graph.encode_point_sequence_with_ranking_and_cost(scheduling_sequence,
                                                                                          self.resource_cluster), dtype=np.float32)
            encoder_list.append(task_encode)

        for i in range(int(graph_number / batch_size)):
            start_batch_index = i * batch_size
            end_batch_index = (i + 1) * batch_size

            task_encode_batch = encoder_list[start_batch_index:end_batch_index]
            if time_major:
                task_encode_batch = np.array(task_encode_batch).swapaxes(0, 1)
                sequence_length = np.asarray([task_encode_batch.shape[0]] * task_encode_batch.shape[1])
            else:
                task_encode_batch = np.array(task_encode_batch)
                sequence_length = np.asarray([task_encode_batch.shape[1]] * task_encode_batch.shape[0])

            decoder_full_lengths.append(sequence_length)
            encoder_lengths.append(sequence_length)
            encoder_batchs.append(task_encode_batch)

            task_graph_batch = task_graph_list[start_batch_index:end_batch_index]
            task_graph_batchs.append(task_graph_batch)
            max_running_time_batchs.append(max_running_time_vector[start_batch_index:end_batch_index])
            min_running_time_batchs.append(min_running_time_vector[start_batch_index:end_batch_index])

            """
            max/min_reliability_batchs update
            """
            max_reliability_batchs.append(max_reliability_vector[start_batch_index:end_batch_index])
            min_reliability_batchs.append(min_reliability_vector[start_batch_index:end_batch_index])

        return encoder_batchs, encoder_lengths, task_graph_batchs, \
               decoder_full_lengths, max_running_time_batchs, \
               min_running_time_batchs, max_reliability_batchs, min_reliability_batchs

    """
    change in calculate_max_min_runningcost
    """
    def calculate_max_min_runningcost(self, max_data_size, min_data_size):
        max_time = max( [self.resource_cluster.up_transmission_cost(max_data_size),
                         self.resource_cluster.dl_transmission_cost(max_data_size),
                         self.resource_cluster.locally_execution_cost(max_data_size)] )

        min_time = self.resource_cluster.mec_execution_cost(min_data_size)

        """
        max_reliability
        min_reliability
        reliability_local = 1
        reliability_mec < 1
        """
        max_reliability = max([math.exp(-self.resource_cluster.up_transmission_cost(max_data_size) * self.failure_rate_mec),
                        math.exp(-self.resource_cluster.dl_transmission_cost(max_data_size) * self.failure_rate_mec),
                        math.exp(-self.resource_cluster.locally_execution_cost(max_data_size) * self.failure_rate_local)])

        min_reliability = math.exp(-self.resource_cluster.mec_execution_cost(min_data_size) * self.failure_rate_mec)

        return max_time, min_time, max_reliability, min_reliability

    """
    change in get_scheduling_cost_step_by_step: reliability computation added + wr_available_time
    """

    def get_scheduling_cost_step_by_step(self, plan, task_graph):
        cloud_avaliable_time = 0.0
        ws_avaliable_time =0.0
        local_avaliable_time = 0.0

        """
        downlink available time
        """
        wr_available_time = 0.0

        # running time on local processor
        T_l = [0] * task_graph.task_number
        # running time on sending channel
        T_ul = [0] * task_graph.task_number
        #running time on receiving channel
        T_dl = [0] * task_graph.task_number


        # finish time on cloud for each task
        FT_cloud = [0] * task_graph.task_number
        # finish time on sending channel for each task
        FT_ws = [0] * task_graph.task_number
        # finish time locally for each task
        FT_locally = [0] * task_graph.task_number
        # finish time recieving channel for each task
        FT_wr = [0] * task_graph.task_number
        current_FT = 0.0
        total_energy = 0.0

        """
        total_reliability added
        """
        total_reliability = 1.0

        return_latency = []
        return_energy = []

        for item in plan:
            i = item[0]
            task = task_graph.task_list[i]
            x = item[1]

            # locally scheduling
            if x == 0:
                if len(task_graph.pre_task_sets[i]) != 0:
                    start_time = max(local_avaliable_time,
                                     max([max(FT_locally[j], FT_wr[j]) for j in task_graph.pre_task_sets[i]]))
                else:
                    start_time = local_avaliable_time

                T_l[i] = self.resource_cluster.locally_execution_cost(task.processing_data_size)
                FT_locally[i] = start_time + T_l[i]
                local_avaliable_time = FT_locally[i]

                task_finish_time = FT_locally[i]

                # calculate the energy consumption
                #energy_consumption = T_l[i] * self.rho * (self.f_l ** self.zeta)

                """
                calculate reliability: local execution
                """
                reliability = math.exp(-T_l[i] * self.failure_rate_local)

            # mcc scheduling
            else:
                """
                FT_wr[j] for parents
                """
                if len(task_graph.pre_task_sets[i]) != 0:
                    ws_start_time = max(ws_avaliable_time,
                                        max([max(FT_locally[j], FT_wr[j])  for j in task_graph.pre_task_sets[i]]))

                    T_ul[i] = self.resource_cluster.up_transmission_cost(task.processing_data_size)
                    ws_finish_time = ws_start_time + T_ul[i]
                    FT_ws[i] = ws_finish_time
                    ws_avaliable_time = ws_finish_time

                    cloud_start_time = max( cloud_avaliable_time,
                                            max([max(FT_ws[i], FT_cloud[j]) for j in task_graph.pre_task_sets[i]]))
                    cloud_finish_time = cloud_start_time + self.resource_cluster.mec_execution_cost(task.processing_data_size)
                    FT_cloud[i] = cloud_finish_time
                    # print("task {}, Cloud finish time {}".format(i, FT_cloud[i]))
                    cloud_avaliable_time = cloud_finish_time

                    """
                    wr_start_time = max(FT_cloud[i], wr_available_time)
                    """
                    wr_start_time = max(wr_available_time, FT_cloud[i])
                    T_dl[i] = self.resource_cluster.dl_transmission_cost(task.transmission_data_size)
                    wr_finish_time = wr_start_time + T_dl[i]
                    FT_wr[i] = wr_finish_time

                    """
                    update wr_available_time
                    """
                    wr_available_time = wr_finish_time

                    # calculate the energy consumption
                    #energy_consumption = T_ul[i] * self.ptx + T_dl[i] * self.prx

                else:
                    ws_start_time = ws_avaliable_time
                    T_ul[i] = self.resource_cluster.up_transmission_cost(task.processing_data_size)
                    ws_finish_time = ws_start_time + T_ul[i]
                    FT_ws[i] = ws_finish_time

                    """
                    update ws_available_time
                    """
                    ws_avaliable_time = ws_finish_time

                    cloud_start_time = max(cloud_avaliable_time, FT_ws[i])
                    cloud_finish_time = cloud_start_time + self.resource_cluster.mec_execution_cost(task.processing_data_size)
                    FT_cloud[i] = cloud_finish_time
                    cloud_avaliable_time = cloud_finish_time

                    """
                    wr_start_time = max(wr_available_time, FT_cloud[i])
                    """
                    wr_start_time = max(wr_available_time, FT_cloud[i])
                    T_dl[i] = self.resource_cluster.dl_transmission_cost(task.transmission_data_size)
                    wr_finish_time = wr_start_time + T_dl[i]
                    FT_wr[i] = wr_finish_time
                    """
                    update wr_available_time
                    """
                    wr_available_time = wr_finish_time

                    # calculate the energy consumption
                    #energy_consumption = T_ul[i] * self.ptx + T_dl[i] * self.prx

                task_finish_time = wr_finish_time

                """
                calculate reliability: mec execution
                """
                reliability = math.exp(-self.failure_rate_mec * (T_ul[i] + T_dl[i] + self.resource_cluster.mec_execution_cost(task.processing_data_size)))

            # print("task  {} finish time is {}".format(i , task_finish_time))
            delta_make_span = max(task_finish_time, current_FT) - current_FT
            current_FT = max(task_finish_time, current_FT)
            return_latency.append(delta_make_span)

            """
            total_reliability update
            """
            total_reliability *= reliability

        return return_latency, current_FT, total_reliability


    def score_func(self, cost, max_time, min_time):
        return -(cost - min_time) / (max_time - min_time)

    """
    change in get_reward_batch_step_by_step: reliability added
    """

    def get_reward_batch_step_by_step(self, action_sequence_batch, task_graph_batch,
                                      max_running_time_batch, min_running_time_batch,
                                      max_reliability_batch, min_reliability_batch):
        target_batch = []
        task_finish_time_batch = []

        """
        total_reliability_batch definition
        """
        total_reliability_batch = []

        for i in range(len(action_sequence_batch)):
            max_running_time = max_running_time_batch[i]
            min_running_time = min_running_time_batch[i]

            task_graph = task_graph_batch[i]
            self.resource_cluster.reset()
            plan = action_sequence_batch[i]
            cost, task_finish_time, total_reliability = self.get_scheduling_cost_step_by_step(plan, task_graph)

            latency = self.score_func(cost, max_running_time, min_running_time)

            score =  np.array(latency)
            #print("score is", score)
            target_batch.append(score)
            task_finish_time_batch.append(task_finish_time)

            total_reliability_batch.append(total_reliability)

        target_batch = np.array(target_batch)
        return target_batch, task_finish_time_batch, total_reliability_batch

    """
    change in greedy_solution: wr_available/start_time changed + reliability computation
    """

    def greedy_solution(self):
        result_plan = []
        finish_time_batchs = []
        """
        reliability_batchs = []
        """
        reliability_batchs = []
        for task_graph_batch in self.task_graphs_batchs:
            plan_batchs = []
            finish_time_plan = []
            """
            reliability_plan = []
            """
            reliability_plan = []
            for task_graph in task_graph_batch:
                cloud_avaliable_time = 0.0
                ws_avaliable_time = 0.0
                local_avaliable_time = 0.0
                """
                wr_available_time
                """
                wr_available_time = 0.0

                """
                reliability initialization
                """
                total_reliability = 1.0

                # finish time on cloud for each task
                FT_cloud = [0] * task_graph.task_number
                # finish time on sending channel for each task
                FT_ws = [0] * task_graph.task_number
                # finish time locally for each task
                FT_locally = [0] * task_graph.task_number
                # finish time recieving channel for each task
                FT_wr = [0] * task_graph.task_number
                plan = []

                for i in task_graph.prioritize_sequence:
                    task = task_graph.task_list[i]

                    # calculate the local finish time
                    if len(task_graph.pre_task_sets[i]) != 0:
                        start_time = max(local_avaliable_time,
                                         max([max(FT_locally[j], FT_wr[j]) for j in task_graph.pre_task_sets[i]]))
                    else:
                        start_time = local_avaliable_time

                    local_running_time = self.resource_cluster.locally_execution_cost(task.processing_data_size)
                    FT_locally[i] = start_time + local_running_time

                    # calculate the remote finish time
                    if len(task_graph.pre_task_sets[i]) != 0:
                        """
                        FT_wr[j] for parents
                        """
                        ws_start_time = max(ws_avaliable_time,
                                            max([max(FT_locally[j], FT_wr[j]) for j in task_graph.pre_task_sets[i]]))
                        FT_ws[i] = ws_start_time + self.resource_cluster.up_transmission_cost(task.processing_data_size)
                        cloud_start_time = max(cloud_avaliable_time,
                                               max([max(FT_ws[i], FT_cloud[j]) for j in task_graph.pre_task_sets[i]]))
                        cloud_finish_time = cloud_start_time + self.resource_cluster.mec_execution_cost(
                            task.processing_data_size)
                        FT_cloud[i] = cloud_finish_time
                        # print("task {}, Cloud finish time {}".format(i, FT_cloud[i]))
                        """
                        wr_start_time = max(wr_available_time, FT_cloud[i])
                        """
                        wr_start_time = max(wr_available_time, FT_cloud[i])
                        wr_finish_time = wr_start_time + self.resource_cluster.dl_transmission_cost(task.transmission_data_size)
                        FT_wr[i] = wr_finish_time
                    else:
                        ws_start_time = ws_avaliable_time
                        ws_finish_time = ws_start_time + self.resource_cluster.up_transmission_cost(task.processing_data_size)
                        FT_ws[i] = ws_finish_time

                        cloud_start_time = max(cloud_avaliable_time, FT_ws[i])
                        FT_cloud[i] = cloud_start_time + self.resource_cluster.mec_execution_cost(
                            task.processing_data_size)
                        """
                        wr_start_time = max(wr_available_time, FT_cloud[i])
                        """
                        FT_wr[i] = max(wr_available_time, FT_cloud[i]) + self.resource_cluster.dl_transmission_cost(task.transmission_data_size)

                    if FT_locally[i] < FT_wr[i]:
                        action = 0
                        local_avaliable_time = FT_locally[i]
                        FT_wr[i] = 0.0
                        FT_cloud[i] = 0.0
                        FT_ws[i] = 0.0

                        """
                        local reliability
                        """
                        reliability = math.exp(-self.failure_rate_local * local_running_time)
                    else:
                        action = 1
                        FT_locally[i] = 0.0
                        cloud_avaliable_time = FT_cloud[i]
                        ws_avaliable_time = FT_ws[i]
                        """
                        update wr_available_time
                        """
                        wr_available_time = FT_wr[i]

                        """
                        mec reliability
                        """
                        reliability = math.exp(-self.failure_rate_mec * (self.resource_cluster.up_transmission_cost(task.processing_data_size)+ self.resource_cluster.mec_execution_cost(
                            task.processing_data_size) + self.resource_cluster.dl_transmission_cost(task.transmission_data_size)))
                    plan.append((i, action))

                    total_reliability *= reliability

                finish_time = max( max(FT_wr), max(FT_locally) )
                plan_batchs.append(plan)
                finish_time_plan.append(finish_time)

                reliability_plan.append(total_reliability)

            finish_time_batchs.append(finish_time_plan)
            result_plan.append(plan_batchs)

            reliability_batchs.append(reliability_plan)

        return result_plan, finish_time_batchs, reliability_batchs

    """
    change in calculate_optimal_solution: reliability sampling
    """

    def calculate_optimal_solution(self):
        # Finding the optimal solution via exhausting search the solution space.
        def exhaustion_plans(n):
            plan_batch = []

            for i in range(2**n):
                plan_str = bin(i)
                plan = []

                for x in plan_str[2:]:
                    plan.append(int(x))

                while len(plan) < n:
                    plan.insert(0, 0)
                plan_batch.append(plan)
            return plan_batch

        n = self.task_graphs_batchs[0][0].task_number
        plan_batch = exhaustion_plans(n)

        print("exhausted plan size: ", len(plan_batch))

        task_graph_optimal_costs = []
        optimal_plan = []

        for task_graph_batch in self.task_graphs_batchs:
            task_graph_batch_cost = []
            for task_graph in task_graph_batch:
                plans_costs = []
                prioritize_plan = []

                plans_rels = []

                for plan in plan_batch:
                    plan_sequence = []
                    for action, task_id in zip(plan, task_graph.prioritize_sequence):
                        plan_sequence.append((task_id, action))

                    cos, task_finish_time, task_reliability = self.get_scheduling_cost_step_by_step(plan_sequence, task_graph)
                    plans_costs.append(task_finish_time)

                    plans_rels.append(task_reliability)

                    prioritize_plan.append(plan_sequence)

                graph_min_cost = min(plans_costs)

                optimal_plan.append(prioritize_plan[np.argmin(plans_costs)])

                task_graph_batch_cost.append(graph_min_cost)

            print("task_graph_batch cost shape is {}".format(np.array(task_graph_batch_cost).shape))
            avg_minimal_cost = np.mean(task_graph_batch_cost)

            task_graph_optimal_costs.append(avg_minimal_cost)

        self.optimal_solution = task_graph_optimal_costs
        return task_graph_optimal_costs, optimal_plan

    """
    change in get_running_cost: reliability sampling
    """

    def get_running_cost(self, action_sequence_batch, task_graph_batch):
        cost_batch = []
        energy_batch = []

        reliability_batch = []

        for action_sequence, task_graph in zip(action_sequence_batch,
                                               task_graph_batch):
            plan_sequence = []

            for action, task_id in zip(action_sequence,
                                       task_graph.prioritize_sequence):
                plan_sequence.append((task_id, action))

                _, task_finish_time, task_reliability = self.get_scheduling_cost_step_by_step(plan_sequence, task_graph)

            cost_batch.append(task_finish_time)

            reliability_batch.append(task_reliability)

        return cost_batch, reliability_batch

    """
    change in get_all_locally_execute_time: reliability sampling
    """
    def get_all_locally_execute_time(self):
        running_cost = []

        running_rel = []
        for task_graph_batch, encode_batch in zip(self.task_graphs_batchs, self.encoder_batchs):
            batch_size = encode_batch.shape[0]
            sequence_length = encode_batch.shape[1]

            scheduling_action = np.zeros(shape=(batch_size, sequence_length), dtype=np.int32)
            running_cost_batch, reliability_batch = self.get_running_cost(scheduling_action, task_graph_batch)
            running_cost.append(np.mean(running_cost_batch))

            running_rel.append((np.mean(reliability_batch)))
        return running_cost, running_rel

    """
    change in get_all_mec_execute_time: reliability sampling    
    """
    def get_all_mec_execute_time(self):
        running_cost = []

        running_rel = []
        for task_graph_batch, encode_batch in zip(self.task_graphs_batchs, self.encoder_batchs):
            batch_size = encode_batch.shape[0]
            sequence_length = encode_batch.shape[1]

            scheduling_action = np.ones(shape=(batch_size, sequence_length), dtype=np.int32)
            running_cost_batch, reliability_batch = self.get_running_cost(scheduling_action, task_graph_batch)

            running_cost.append(np.mean(running_cost_batch))

            running_rel.append(np.mean(reliability_batch))
        return running_cost, running_rel

    def greedy_solution_for_current_task(self):
        result_plan, finish_time_batchs, reliability_batchs = self.greedy_solution()

        return result_plan[self.task_id], finish_time_batchs[self.task_id], reliability_batchs[self.task_id]



