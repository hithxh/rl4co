import math
import numpy as np

class Mask():
    def __init__(self, tasks):
        self.tasks = tasks
    def cal_angle(self,point1, point2):
        # 计算两点之间的角度
        delta_x = point2[0] - point1[0]
        delta_y = point2[1] - point1[1]
        angle = math.atan2(delta_y, delta_x) * 180 / math.pi
        return angle

    def mask_matrix(self):
        '''Calculate angle between tasks and mask matrix.
        Args:
            tasks (np.array): (batch_size, n_node, 2)
        Returns:
            angle (np.array): (batch_size, n_node, n_node)
            '''
        batch_size, n_node, _ = self.tasks.shape
        angle = np.zeros((batch_size, n_node, n_node))
        mask_matrix = np.zeros((batch_size, n_node, n_node))
        for i in range(batch_size):
            for j in range(n_node):
                for k in range(n_node):
                    angle[i,j,k] = self.cal_angle(self.tasks[i,j], self.tasks[i,k])
                    
                    if -30 < angle[i,j,k] < 30:
                        mask_matrix[i,j,k] = 1
        return mask_matrix

