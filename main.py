import numpy as np
import math
from dataclasses import dataclass
from typing import List, Optional, Dict
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.spatial import cKDTree as KDTree


@dataclass
class Pose2D:
    """Represents a 2D robot pose (x, y, θ) in SE(2)"""
    x: float
    y: float
    theta: float = 0.0

    def to_matrix(self) -> np.ndarray:
        """Pose2D to Homogeneous transformation matrix (3x3 for SE2)"""
        c = math.cos(self.theta)
        s = math.sin(self.theta)
        return np.array([
            [c, -s, self.x],
            [s,  c, self.y],
            [0,  0,    1]
        ])

    def inverse(self) -> 'Pose2D':
        """Inverse transformation T_inv"""
        c = math.cos(self.theta)
        s = math.sin(self.theta)
        inv_x = -c * self.x - s * self.y
        inv_y = s * self.x - c * self.y
        return Pose2D(inv_x, inv_y, -self.theta)

    def compose(self, other: 'Pose2D') -> 'Pose2D':
        """T_AB = T_A * T_B"""
        T_self = self.to_matrix()
        T_other = other.to_matrix()
        T_composed = T_self @ T_other
        
        x = T_composed[0, 2]
        y = T_composed[1, 2]
        theta = math.atan2(T_composed[1, 0], T_composed[0, 0])
        return Pose2D(x, y, theta)

    @staticmethod
    def from_matrix(matrix: np.ndarray) -> 'Pose2D':
        """Homogeneous transformation matrix to Pose2D"""
        x = matrix[0, 2]
        y = matrix[1, 2]
        theta = math.atan2(matrix[1, 0], matrix[0, 0])
        return Pose2D(x, y, theta)

    def adjoint(self) -> np.ndarray:
        """Ad(T) in SE(2)"""
        c = math.cos(self.theta)
        s = math.sin(self.theta)
        
        return np.array([
            [c, -s,  self.y],
            [s,  c, -self.x],
            [0,  0,  1]
        ])

    @staticmethod
    def exp(xi: np.ndarray) -> 'Pose2D':
        """
        se(2) (twist vector xi) to SE(2) (Pose2D)
        xi = [dx, dy, dtheta]
        """
        dx, dy, dtheta = xi

        if np.isclose(dtheta, 0.0, atol=1e-9): # Pure translation (theta near zero)
            return Pose2D(dx, dy, 0.0)
        else:
            c = np.cos(dtheta)
            s = np.sin(dtheta)
            
            # The translation part of the exp map: J(w) * [vx, vy]^T
            # where J(w) = [sin(w)/w, (1-cos(w))/w; -(1-cos(w))/w, sin(w)/w]
            A = s / dtheta
            B = (1 - c) / dtheta
            
            trans_x = A * dx + B * dy
            trans_y = -B * dx + A * dy
            
            return Pose2D(trans_x, trans_y, dtheta)

    def log(self) -> np.ndarray:
        """
        SE(2) (Pose2D) to se(2) (twist vector xi)
        Returns [dx, dy, dtheta]
        """
        x, y, theta = self.x, self.y, self.theta
        
        if np.isclose(theta, 0.0, atol=1e-9): # Pure translation
            return np.array([x, y, 0.0])
        else:
            half_theta = theta / 2.0
            
            # Special case for theta near pi (180 degrees) due to numerical issues with tan(theta/2)
            if np.isclose(abs(theta), np.pi, atol=1e-9):
                vx = y / np.pi
                vy = -x / np.pi
            else:
                cot_half_theta = 1.0 / np.tan(half_theta)
                
                term1 = half_theta * cot_half_theta
                term2 = half_theta
                
                vx = term1 * x + term2 * y
                vy = -term2 * x + term1 * y
            
            return np.array([vx, vy, theta])

    def __add__(self, delta: np.ndarray) -> 'Pose2D':
        """
        Apply a delta vector (from se(2) / twist) to the pose (SE(2)) using exponential map.
        This is the correct Lie Group update: T_new = T_old * exp(delta_twist)
        """
        delta_pose = Pose2D.exp(delta)
        return self.compose(delta_pose)

    def __sub__(self, other: 'Pose2D') -> np.ndarray:
        """
        Compute the 'error' in se(2) (twist vector) from 'other' to 'self'.
        This represents the delta to go from 'other' to 'self'.
        error = log(other_inv * self)
        """
        relative_transform_matrix = other.inverse().to_matrix() @ self.to_matrix()
        relative_transform_pose = Pose2D.from_matrix(relative_transform_matrix)
        return relative_transform_pose.log()


@dataclass
class Edge:
    """Represents a constraint (edge) in the pose graph"""
    from_id: int
    to_id: int
    measurement: Pose2D # The observed relative transformation
    information: np.ndarray # Information matrix (inverse of covariance)
    edge_type: str = "odometry" # "odometry", "observation", "loop_closure"


class PoseGraph:
    def __init__(self):
        self.nodes: List[Pose2D] = []
        self.edges: List[Edge] = []
        self.node_covariances: Dict[int, np.ndarray] = {} 
        self.node_id_to_idx: Dict[int, int] = {} 


    def add_node(self, pose: Pose2D, 
                 initial_covariance: Optional[np.ndarray] = None) -> int:
        """Add a new node (robot pose) to the graph"""
        node_id = len(self.nodes)
        self.nodes.append(pose)
        self.node_id_to_idx[node_id] = node_id * 3 
        if initial_covariance is not None:
            self.node_covariances[node_id] = initial_covariance
        return node_id

    def add_edge(self, from_id: int, to_id: int, 
                 measurement: Pose2D, information: np.ndarray, 
                 edge_type: str = "odometry") -> None:
        """Add a new edge (constraint) to the graph"""
        edge = Edge(from_id, to_id, measurement, information, edge_type)
        self.edges.append(edge)

    @property
    def state_dimension(self) -> int:
        """Total dimension of the state vector (all pose parameters)"""
        return len(self.nodes) * 3


# Graph Construction
class GraphBuilder:
    def __init__(self, initial_pose: Pose2D, initial_covariance: np.ndarray):
        self.pose_graph = PoseGraph()
        self.current_pose_id = self.pose_graph.add_node(initial_pose, initial_covariance)

    def add_odometry_measurement(self, 
                                odom_measurement: Pose2D, # Relative pose transformation
                                uncertainty: np.ndarray) -> None:
        prev_pose = self.pose_graph.nodes[self.current_pose_id]
        new_pose = prev_pose.compose(odom_measurement)
        
        new_pose_id = self.pose_graph.add_node(new_pose)
        
        information = np.linalg.inv(uncertainty)
        self.pose_graph.add_edge(
            self.current_pose_id,
            new_pose_id,
            odom_measurement,
            information,
            edge_type="odometry"
        )
        self.current_pose_id = new_pose_id


def normalize_angle(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi

# Optimization (Gauss-Newton)
class GraphOptimizer:
    def __init__(self, pose_graph: PoseGraph):
        self.pose_graph = pose_graph

    def optimize(self, max_iterations: int = 100, 
                 convergence_thresh: float = 1e-6) -> None:
        prev_total_error = float('inf') 

        for iteration in range(max_iterations):
            current_total_error = self.compute_total_error()
            print(f"Iteration {iteration}: Current error: {current_total_error:.4f}")

            # Detect Convergence
            if abs(prev_total_error - current_total_error) < convergence_thresh:
                print(f"Converged after {iteration} iterations. Final error: {current_total_error:.4f}")
                break

            # Stop when significantly diverging
            if current_total_error > prev_total_error * 10 and iteration > 0:
                print(f"Warning: Error diverging. Stopping optimization. Current error: {current_total_error:.4f}")
                break

            if np.isnan(current_total_error):
                print(f"Error is NaN. Stopping optimization.")
                break

            prev_total_error = current_total_error

            H = sparse.lil_matrix((self.pose_graph.state_dimension, self.pose_graph.state_dimension))
            b = np.zeros(self.pose_graph.state_dimension)

            H[0:3, 0:3] += np.eye(3) * 1e9 

            for edge in self.pose_graph.edges:
                from_idx_start = self.pose_graph.node_id_to_idx[edge.from_id]
                to_idx_start = self.pose_graph.node_id_to_idx[edge.to_id]
                
                pose_i = self.pose_graph.nodes[edge.from_id]
                pose_j = self.pose_graph.nodes[edge.to_id]
                
                Z_ij = edge.measurement
                
                T_ij_est = pose_i.inverse().compose(pose_j)

                error_vector = Z_ij.inverse().compose(T_ij_est).log()
                
                omega = edge.information 

                J_i = - edge.measurement.inverse().adjoint()
                J_j = np.eye(3) 

                H_ii = J_i.T @ omega @ J_i
                H_ij = J_i.T @ omega @ J_j
                H_jj = J_j.T @ omega @ J_j

                H[from_idx_start:from_idx_start+3, from_idx_start:from_idx_start+3] += H_ii
                H[from_idx_start:from_idx_start+3, to_idx_start:to_idx_start+3] += H_ij
                H[to_idx_start:to_idx_start+3, from_idx_start:from_idx_start+3] += H_ij.T 
                H[to_idx_start:to_idx_start+3, to_idx_start:to_idx_start+3] += H_jj
                
                b[from_idx_start:from_idx_start+3] += J_i.T @ omega @ error_vector
                b[to_idx_start:to_idx_start+3] += J_j.T @ omega @ error_vector

            H_sparse = H.tocsr()
            
            try:
                delta_x = sparse.linalg.spsolve(H_sparse, -b)
            except RuntimeError as e:
                print(f"Warning: Sparse solver error - {e}. Falling back to dense solve.")
                try:
                    delta_x = np.linalg.solve(H_sparse.todense(), -b)
                except np.linalg.LinAlgError:
                    print("ERROR: Singular matrix encountered during dense solve. Optimization stopped.")
                    break
            
            if np.linalg.norm(delta_x) < convergence_thresh * 1e-2:
                 print(f"Delta_x norm ({np.linalg.norm(delta_x):.6e}) below threshold. Converged.")
                 break

            for i in range(len(self.pose_graph.nodes)):
                idx_start = self.pose_graph.node_id_to_idx[i]
                delta_pose_vector = delta_x[idx_start:idx_start+3]

                delta_pose_vector[2] = normalize_angle(delta_pose_vector[2])
                
                self.pose_graph.nodes[i] = self.pose_graph.nodes[i] + delta_pose_vector
            
        else:
            print(f"Optimization finished after {max_iterations} iterations. Final error: {self.compute_total_error():.4f}")

    def compute_total_error(self) -> float:
        total_error = 0.0
        
        for edge in self.pose_graph.edges:
            pose_i = self.pose_graph.nodes[edge.from_id]
            pose_j = self.pose_graph.nodes[edge.to_id]
            
            T_ij_est = pose_i.inverse().compose(pose_j)
            
            error_vector = edge.measurement.inverse().compose(T_ij_est).log()
            weighted_error = error_vector.T @ edge.information @ error_vector
            total_error += weighted_error
        
        fixed_node_error_vector = self.pose_graph.nodes[0].log()
        prior_information = np.eye(3) * 1e9 
        total_error += fixed_node_error_vector.T @ prior_information @ fixed_node_error_vector

        return total_error


# Visualization
class SLAMVisualizer:
    def __init__(self, pose_graph: PoseGraph, ax=None):
        self.pose_graph = pose_graph
        if ax is None:
            self.fig, self.ax = plt.subplots(figsize=(6, 6))
        else:
            self.ax = ax

        self.ax.set_aspect('equal', adjustable='box')

    def visualize_graph(self, title: str = "SLAM Graph") -> None:
        self.ax.clear()
        self.ax.set_title(title)

        poses_x = [pose.x for pose in self.pose_graph.nodes]
        poses_y = [pose.y for pose in self.pose_graph.nodes]
        self.ax.plot(poses_x, poses_y, 'b.-', label='Robot Path')
        
        for edge in self.pose_graph.edges:
            p1 = self.pose_graph.nodes[edge.from_id]
            p2 = self.pose_graph.nodes[edge.to_id]
            
            line_style = '-'
            alpha = 0.7
            if edge.edge_type == "odometry":
                color = 'g'
            elif edge.edge_type == "loop_closure":
                color = 'r'
                line_style = '--'
            else:
                color = 'y'
            
            self.ax.plot([p1.x, p2.x], [p1.y, p2.y], color=color, linestyle=line_style, alpha=alpha, linewidth=0.5)

        # orientation arrows
        # for pose in self.pose_graph.nodes:
        #     dx = 0.2 * math.cos(pose.theta)
        #     dy = 0.2 * math.sin(pose.theta)
        #     self.ax.arrow(pose.x, pose.y, dx, dy, head_width=0.05, fc='k', ec='k')
        
        self.ax.legend()
        self.ax.set_xlabel("X Position (m)")
        self.ax.set_ylabel("Y Position (m)")
        self.ax.grid(True)


def run_slam_demo():
    print("Starting Graph SLAM Demonstration...")

    initial_pose = Pose2D(0, 0, 0)
    initial_covariance = np.diag([0.01, 0.01, np.radians(1)])**2 

    graph_builder = GraphBuilder(initial_pose, initial_covariance)
    
    odom_uncertainty = np.diag([0.1, 0.1, np.radians(5)])**2 

    # Simulated Path with noise
    print("\nSimulating robot movement and adding odometry...")
    path_segments = [
        (Pose2D(1, 0, 0), 10), 
        (Pose2D(0, 0, np.radians(90)), 5), 
        (Pose2D(1, 0, 0), 10), 
        (Pose2D(0, 0, np.radians(90)), 5), 
        (Pose2D(1, 0, 0), 10), 
        (Pose2D(0, 0, np.radians(90)), 5), 
        (Pose2D(1, 0, 0), 10), 
        (Pose2D(0, 0, np.radians(90)), 6), 
    ]
    
    np.random.seed(2)
    odom_noise_scale = 0.05

    for rel_pose_template, steps in path_segments:
        for _ in range(steps):
            noisy_rel_pose = Pose2D(
                rel_pose_template.x + np.random.normal(0, odom_noise_scale),
                rel_pose_template.y + np.random.normal(0, odom_noise_scale),
                rel_pose_template.theta + np.random.normal(0, odom_noise_scale / 5.0) 
            )
            graph_builder.add_odometry_measurement(noisy_rel_pose, odom_uncertainty)

    print(f"Initial graph built with {len(graph_builder.pose_graph.nodes)} poses and "
          f"{len(graph_builder.pose_graph.edges)} odometry edges.")

    # Plot graph
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)

    initial_visualizer = SLAMVisualizer(graph_builder.pose_graph, ax=ax1)
    initial_visualizer.visualize_graph("Initial Path (with Odometry Drift)")

    # After SLAM corrected graph
    print("\nAdding a simulated loop closure constraint...")
    
    loop_closure_measurement = Pose2D(0.1, -0.2, np.radians(5)) 
    loop_closure_information = np.diag([1.0, 1.0, 0.1])**2 * 10
    
    graph_builder.pose_graph.add_edge(
        graph_builder.current_pose_id, 
        0,                             
        loop_closure_measurement,      
        loop_closure_information,
        edge_type="loop_closure"
    )
    print(f"Added loop closure edge from node {graph_builder.current_pose_id} to node 0.")
    
    initial_visualizer.visualize_graph("Path with Loop Closure (Before Optimization)")

    print("\nStarting Graph Optimization (Gauss-Newton)...")
    optimizer = GraphOptimizer(graph_builder.pose_graph)
    optimizer.optimize(max_iterations=50)

    # Plot graph
    optimized_visualizer = SLAMVisualizer(graph_builder.pose_graph, ax=ax2)
    optimized_visualizer.visualize_graph("Optimized Path")

    # Axes
    all_x = [pose.x for pose in graph_builder.pose_graph.nodes]
    all_y = [pose.y for pose in graph_builder.pose_graph.nodes]
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)

    ax1.set_xlim(x_min - 1, x_max + 1)
    ax1.set_ylim(y_min - 1, y_max + 1)
    ax2.set_xlim(x_min - 1, x_max + 1)
    ax2.set_ylim(y_min - 1, y_max + 1)

    plt.show()

    print("\nDemo Complete!")
    print("Close the plot windows to exit.\n")


if __name__ == "__main__":
    run_slam_demo()