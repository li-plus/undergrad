import os
import argparse


class Cpu(object):
    STATE_RUNNING = 1
    STATE_IDLE = 2
    def __init__(self):
        self.state = self.STATE_IDLE
        self.curr_jobname = None

    def __repr__(self):
        return 'Cpu(state: {}, curr_jobname: ,{})'.format(self.state, self.curr_jobname)


class Job(object):
    def __init__(self, name, runtime, prio):
        self.name = name
        self.runtime = runtime
        self.prio = prio

    def __repr__(self):
        return 'Jobs(name: {}, runtime: {}, prio: {})'.format(self.name, self.runtime, self.prio)


def run_o1_scheduler(jobs, cpus, time_slice, max_prio):
    runqueue = [[] for _ in range(max_prio)]
    expire_queue = [[] for _ in range(max_prio)]
    active_bitarray = [0 for _ in range(max_prio)]
    expire_bitarray = [0 for _ in range(max_prio)]

    # init expire bitarray & queue
    for job_name, job in jobs.items():
        assert 0 <= job.prio < max_prio
        expire_queue[job.prio].append(job_name)
        expire_bitarray[job.prio] = 1

    system_time = 0
    jobs_finished = 0

    while jobs_finished < len(jobs):
        # handle interrupts
        if system_time % time_slice == 0:
            for cpu in cpus:
                if cpu.state == Cpu.STATE_RUNNING:
                    curr_job = jobs[cpu.curr_jobname]
                    if curr_job.runtime > 0:
                        expire_queue[curr_job.prio].append(curr_job.name)
                        expire_bitarray[curr_job.prio] = 1
                    cpu.state = Cpu.STATE_IDLE
                    cpu.curr_jobname = None

        # assign a job for each cpu
        for cpu in cpus:
            if cpu.state == Cpu.STATE_IDLE:
                if 1 not in active_bitarray:
                    active_bitarray, expire_bitarray = expire_bitarray, active_bitarray
                    runqueue, expire_queue = expire_queue, runqueue

                if 1 not in active_bitarray:
                    break

                prio = active_bitarray.index(1)
                job_name = runqueue[prio].pop(0)
                if not runqueue[prio]:
                    active_bitarray[prio] = 0

                cpu.state = Cpu.STATE_RUNNING
                cpu.curr_jobname = job_name

        # run each cpu for one tick
        for cpu in cpus:
            if cpu.state == Cpu.STATE_RUNNING:
                job = jobs[cpu.curr_jobname]
                job.runtime -= 1

                if job.runtime <= 0:
                    cpu.state = Cpu.STATE_IDLE
                    cpu.curr_jobname = None
                    jobs_finished += 1

        # trace
        if system_time % time_slice == 0:
            print('-' * (7 + len(cpus) * 17))
        print('{:4d}:'.format(system_time), end='  ')
        for cpu_index, cpu in enumerate(cpus):
            if cpu.state == Cpu.STATE_RUNNING:
                print('CPU {:d}: {:2s} [{:3d}]'.format(cpu_index, cpu.curr_jobname, jobs[cpu.curr_jobname].runtime), end='  ')
            else:
                print('CPU {:d}: -  [   ]'.format(cpu_index), end='  ')
        print('')

        system_time += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-cpus', type=int, default=2, help='How many CPU devices')
    parser.add_argument('--jobs-priority', type=int, nargs='+', default=[1, 2, 3], help='Priorities for each job')
    parser.add_argument('--jobs-runtime', type=int, nargs='+', default=[35, 25, 15], help='Runtimes for each job')
    parser.add_argument('--time-slice', type=int, default=10, help='Length of a time slice')
    args = parser.parse_args()

    jobs = {}
    for job_index, (prio, runtime) in enumerate(zip(args.jobs_priority, args.jobs_runtime)):
        jobs[str(job_index)] = Job(str(job_index), runtime, prio)

    cpus = []
    for _ in range(args.num_cpus):
        cpus.append(Cpu())

    run_o1_scheduler(jobs, cpus, args.time_slice, max(args.jobs_priority) + 1)


if __name__ == '__main__':
    main()
