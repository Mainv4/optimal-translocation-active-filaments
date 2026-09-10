

import argparse
import os

import MDAnalysis as mda
import numpy as np
from tqdm import tqdm


def read_trajectory(path, name_of_trajectory_file, name_of_topology_file):
    print("Reading the trajectory...")
    u = mda.Universe(
        path + "/" + name_of_topology_file,
        path + "/" + name_of_trajectory_file,
        topology_format="DATA",
        format="LAMMPSDUMP",
    )
    print("Trajectory read")
    return u


def calc_MSD(path, u, N):
    print("Calculating the MSD...")
    path_array = path.split("/")
    output_file_name_head = "__".join(path_array[:]) + "MSD__HeadOverTime.dat"
    output_file_name_tail = "__".join(path_array[:]) + "MSD__TailOverTime.dat"
    output_file_name_CM = "__".join(path_array[:]) + "MSD__CMOverTime.dat"

    MSD_head = np.zeros((u.trajectory.n_frames, N))
    MSD_tail = np.zeros((u.trajectory.n_frames, N))
    MSD_CM = np.zeros((u.trajectory.n_frames, N))

    pos_head = np.zeros((u.trajectory.n_frames, N, 3))
    pos_tail = np.zeros((u.trajectory.n_frames, N, 3))
    pos_CM = np.zeros((u.trajectory.n_frames, N, 3))
    for t in range(u.trajectory.n_frames):
        u.trajectory[t]
        for n in range(N):
            head = u.select_atoms("resid " + str(n + 1))[-1]
            tail = u.select_atoms("resid " + str(n + 1))[0]
            CM = u.select_atoms("resid " + str(n + 1)).center_of_mass()
            pos_head[t][n] = head.position
            pos_tail[t][n] = tail.position
            pos_CM[t][n] = CM

    pos_head = pos_head[100:]
    pos_tail = pos_tail[100:]
    pos_CM = pos_CM[100:]
    MSD_head = MSD_head[100:]
    MSD_tail = MSD_tail[100:]
    MSD_CM = MSD_CM[100:]
    for tau in tqdm(range(len(MSD_head))):
        for t in range(len(MSD_head) - tau):
            for n in range(N):
                MSD_head[tau][n] += (
                    np.linalg.norm(pos_head[t + tau][n] - pos_head[t][n]) ** 2
                )
                MSD_tail[tau][n] += (
                    np.linalg.norm(pos_tail[t + tau][n] - pos_tail[t][n]) ** 2
                )
                MSD_CM[tau][n] += np.linalg.norm(pos_CM[t + tau][n] - pos_CM[t][n]) ** 2
    for t in range(len(MSD_head)):
        MSD_head[t] /= len(MSD_head) - t
        MSD_tail[t] /= len(MSD_tail) - t
        MSD_CM[t] /= len(MSD_CM) - t
    np.savetxt("DATA_MSD/" + output_file_name_head, MSD_head, fmt="%f")
    np.savetxt("DATA_MSD/" + output_file_name_tail, MSD_tail, fmt="%f")
    np.savetxt("DATA_MSD/" + output_file_name_CM, MSD_CM, fmt="%f")

    print("MSD calculated")


def calc_MSD_x(path, u, N):
    print("Calculating the MSD...")
    path_array = path.split("/")
    output_file_name_head = "__".join(path_array[:]) + "MSD__HeadOverTime.dat"
    output_file_name_tail = "__".join(path_array[:]) + "MSD__TailOverTime.dat"
    output_file_name_CM = "__".join(path_array[:]) + "MSD__CMOverTime.dat"

    MSD_head = np.zeros((u.trajectory.n_frames, N))
    MSD_tail = np.zeros((u.trajectory.n_frames, N))
    MSD_CM = np.zeros((u.trajectory.n_frames, N))

    pos_head = np.zeros((u.trajectory.n_frames, N, 3))
    pos_tail = np.zeros((u.trajectory.n_frames, N, 3))
    pos_CM = np.zeros((u.trajectory.n_frames, N, 3))
    for t in range(u.trajectory.n_frames):
        u.trajectory[t]
        for n in range(N):
            head = u.select_atoms("resid " + str(n + 1))[-1]
            tail = u.select_atoms("resid " + str(n + 1))[0]
            CM = u.select_atoms("resid " + str(n + 1)).center_of_mass()
            pos_head[t][n] = head.position
            pos_tail[t][n] = tail.position
            pos_CM[t][n] = CM

    pos_head = pos_head[100:]
    pos_tail = pos_tail[100:]
    pos_CM = pos_CM[100:]
    MSD_head = MSD_head[100:]
    MSD_tail = MSD_tail[100:]
    MSD_CM = MSD_CM[100:]
    for tau in tqdm(range(len(MSD_head))):
        for t in range(len(MSD_head) - tau):
            for n in range(N):
                MSD_head[tau][n] += (pos_head[t + tau][n][0] - pos_head[t][n][0]) ** 2
                MSD_tail[tau][n] += (pos_tail[t + tau][n][0] - pos_tail[t][n][0]) ** 2
                MSD_CM[tau][n] += (pos_CM[t + tau][n][0] - pos_CM[t][n][0]) ** 2
    for t in range(len(MSD_head)):
        MSD_head[t] /= len(MSD_head) - t
        MSD_tail[t] /= len(MSD_tail) - t
        MSD_CM[t] /= len(MSD_CM) - t
    np.savetxt("DATA_MSD_x/" + output_file_name_head, MSD_head, fmt="%f")
    np.savetxt("DATA_MSD_x/" + output_file_name_tail, MSD_tail, fmt="%f")
    np.savetxt("DATA_MSD_x/" + output_file_name_CM, MSD_CM, fmt="%f")

    print("MSD calculated")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate the mean square displacement of the CM, the Head, and the Tail polymers."
    )
    parser.add_argument(
        "--directory",
        type=str,
        help="The path to the directory containing the trajectory.",
    )
    parser.add_argument("--topology", type=str, help="The name of the topology file.")
    parser.add_argument(
        "--trajectory", type=str, help="The name of the trajectory file."
    )
    parser.add_argument("-N", type=int, help="The number of polymers.")
    args = parser.parse_args()

    path = args.directory
    topology = args.topology.split("/")[-1]
    trajectory = args.trajectory.split("/")[-1]
    N = args.N

    os.makedirs("DATA_MSD", exist_ok=True)
    os.makedirs("DATA_MSD_x", exist_ok=True)

    u = read_trajectory(path, trajectory, topology)

    calc_MSD(path, u, N)
    calc_MSD_x(path, u, N)
